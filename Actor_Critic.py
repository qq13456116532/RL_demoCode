'''
1. Actor: 负责学习和采取行动，尝试找到最优策略来最大化累积奖励。
2. Critic: 负责评估Actor的行为，并提供反馈来改进Actor的策略。

策略梯度公式（带基线）:
Δθ = α∇θ[log π(θ|s)]*(Q(s,a) - b)
从上面这个公式使用V(S)来代替b，得到：
Δθ = α ∇θ[log π(θ|s)]*(Q(s,a) - V(s))
得到
Δθ = α ∇θ[log π(θ|s)]*A(s)
具体可看 line.69

'''


from torch import nn, optim

import rl_utils
from BaseAgent import BaseAgent
from config import *


# 策略网络PolicyNet（与 REINFORCE 算法一样）
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


class AC_Agent(BaseAgent):
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)
        # 时序差分目标,实际上就是 Q(s,a)
        td_target = rewards + self.reward_gamma * self.critic_net(next_states) * (1 - dones)
        # 计算的是Advantages优势，即Q(s,a) - V(s)
        td_delta = td_target - self.critic_net(states)  # 时序差分误差
        log_probs = torch.log(self.actor_net(states).gather(1, actions))
        # A2C的actor-loss
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            self.critic_loss(self.critic_net(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数

    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, reward_gamma, device):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.device = device
        self.actor_net = Actor(state_dim, action_dim).to(device)
        self.critic_net = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.critic_loss = nn.MSELoss()


if __name__ == "__main__":
    ALG_NAME = 'Actor-Critic'
    print("这是强化学习 " + ALG_NAME + " 算法")
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 1000
    gamma = 0.98
    max_step = 9999
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = AC_Agent(state_dim, action_dim, actor_lr,critic_lr, gamma, DEVICE)
    rl_utils.train_on_policy_agent(env, agent, num_episodes, Name=ALG_NAME, max_episode_size=max_step)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor_net.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic_net.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(ENV_ID, agent)


