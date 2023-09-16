"""
Proximal Policy Optimization是一种高效和稳定的策略优化方法。
它试图找到一个在保持策略更新稳定的同时，能够实现较大步长的优化算法。

它通过引入了一个 裁剪的目标函数 来限制策略更新的步长，从而避免更新过大导致训练不稳定。


PPO采用以下目标函数进行策略更新 ,见 line.88
L(θ) = min(π(a|s;θ)/π(a|s;θ_old) * A(s, a), clip(π(a|s;θ)/π(a|s;θ_old), 1-ε, 1+ε) * A(s, a))
其中，
- θ: 当前策略的参数
- θ_old: 上一次策略更新后的参数
- A(s, a): 动作a在状态s下的优势函数
- ε: 一个小的正数，用来限制策略更新的步长

然后这里的 A(s,a)是使用GAE的方式来计算

通过多次迭代优化这个目标函数来更新策略参数，PPO能够实现在训练过程中保持较高的样本效率和稳定性。

"""





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


class PPOAgent(BaseAgent):
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict,self.device)
        td_target = rewards + self.reward_gamma * self.critic_net(next_states) * (1-dones)
        td_delta = td_target - self.critic_net(states)
        # GAE计算的就是 (td1+(γλ)td2+(γλ)^2td3+(γλ)^3td4+.....,td2+(γλ)td3+(γλ)^2td4+....,td3+(γλ)td4+(γλ)^2td5..... )
        advantage = rl_utils.compute_advantage(self.reward_gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor_net(states).gather(1,actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor_net(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            # π(a|s;θ)/π(a|s;θ_old) * A(s, a)
            surr1 = ratio * advantage
            # clip(π(a|s;θ)/π(a|s;θ_old), 1-ε, 1+ε) * A(s, a)
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            # 取小
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(self.critic_loss(self.critic_net(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()



    def __init__(self, state_dim, action_dim, reward_gamma, device,lmbda,eps, epochs,actor_lr=1e-3,crtic_lr=1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.lmbda =lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.epochs = epochs # 一条序列的数据用来训练轮数
        self.device = device
        self.actor_net = Actor(state_dim, action_dim).to(device)
        self.critic_net = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=crtic_lr)
        self.critic_loss = nn.MSELoss()


if __name__ == "__main__":
    ALG_NAME = 'ppo'
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

    agent = PPOAgent(state_dim, action_dim, gamma, DEVICE,lmbda,eps,epochs,actor_lr,critic_lr)
    rl_utils.train_on_policy_agent(env, agent, num_episodes, Name=ALG_NAME, max_episode_size=MAX_STEPS)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor_net.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic_net.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(ENV_ID, agent)
