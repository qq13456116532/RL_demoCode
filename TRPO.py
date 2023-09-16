"""
已被抛弃的算法，PPO完美替代且更简单
"""



import copy

from torch import nn, optim

import rl_utils
from BaseAgent import BaseAgent
from config import *


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        state = self.relu(self.fc1(state))
        state = self.fc2(state)
        probs = self.softmax(state)
        return probs


# 价值网络ValueNet，其输入是某个状态，输出则是状态的价值
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = self.relu(self.fc1(state))
        value = self.fc2(state)
        return value


class TRPOAgent(BaseAgent):
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def compute_advantage(self, gamma, lmbda, td_delta):
        td_delta = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float)

    def hessian_matrix_vector_product(self, states, old_action_dists, vector):
        # 计算黑塞矩阵和一个向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor_net(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  # 计算平均KL距离
        kl_grad = torch.autograd.grad(kl,
                                      self.actor_net.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL距离的梯度先和向量进行点积运算
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor_net.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    def conjugate_gradient(self, grad, states, old_action_dists):  # 共轭梯度法求解方程
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):  # 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs).to(self.device)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec):  # 线性搜索
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor_net.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor_net)
        for i in range(15):  # 线性搜索主循环
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor_net)
            torch.nn.utils.convert_parameters.vector_to_parameters(
                new_para, new_actor.parameters())
            new_action_dists = torch.distributions.Categorical(
                new_actor(states))
            kl_div = torch.mean(
                torch.distributions.kl.kl_divergence(old_action_dists,
                                                     new_action_dists))
            new_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                 old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict,self.device)

        # # # 时序差分目标
        td_target = rewards + self.reward_gamma * self.critic_net(next_states) * (1 - dones)
        td_delta = td_target - self.critic_net(states)  # 时序差分误差
        advantages = self.compute_advantage(self.reward_gamma, self.lambda_GAE, td_delta).to(self.device)
        old_log_probs = torch.log(self.actor_net(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor_net(states).detach())
        critic_loss = torch.mean(self.critic_loss(self.critic_net(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()  # 更新价值函数

        # 更新策略函数
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantages,
                                                   old_log_probs, self.actor_net)
        grads = torch.autograd.grad(surrogate_obj, self.actor_net.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 用共轭梯度法计算x = H^(-1)g
        descent_direction = self.conjugate_gradient(obj_grad, states,
                                                    old_action_dists)

        Hd = self.hessian_matrix_vector_product(states, old_action_dists,
                                                descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint /
                              (torch.dot(descent_direction, Hd) + 1e-8))
        new_para = self.line_search(states, actions, advantages, old_log_probs,
                                    old_action_dists,
                                    descent_direction * max_coef)  # 线性搜索
        torch.nn.utils.convert_parameters.vector_to_parameters(
            new_para, self.actor_net.parameters())  # 用线性搜索后的参数更新策略

    def __init__(self, state_dim, action_dim, actor_lr,crtic_lr, reward_gamma, device, lambda_GAE):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.device = device
        self.actor_net = Actor(state_dim, action_dim).to(device)
        self.critic_net = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=crtic_lr)
        self.critic_loss = nn.MSELoss()
        self.lambda_GAE = lambda_GAE
        self.alpha = 0.5
        self.kl_constraint = 0.0005


if __name__ == "__main__":
    ALG_NAME = 'TRPO'
    print("这是强化学习 " + ALG_NAME + " 算法")
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    lmbda = 0.95
    epochs = 10
    eps = 0.2
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = TRPOAgent(state_dim, action_dim, actor_lr, critic_lr,gamma, DEVICE, lambda_GAE=0.95)
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    rl_utils.train_on_policy_agent(env, agent, 1000, Name=ALG_NAME, max_episode_size=MAX_STEPS)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor_net.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic_net.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(ENV_ID, agent)
