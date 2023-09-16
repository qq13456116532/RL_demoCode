'''
1. Actor: 负责选择连续的动作，以便根据当前策略最大化预期奖励。
2. Critic: 负责估计Q值（状态-动作函数），这个Q值被用来更新Actor的策略和评估当前策略的好坏。

DDPG算法是一种基于Actor-Critic的方法，但适用于连续动作空间。它结合了DPG（确定性策略梯度）和DQN（深度Q网络）的思想。
其中，Critic学习Q函数来评估每一对状态-动作的预期回报，而Actor则负责生成动作。

策略更新公式（DDPG）：
Δθ = α ∇θ Q(s,a|θ)
   = α * [∇w Q(s,a|w)] * [∇θ A(s|θ)]   ,这是用到链式法则，两个梯度都是对神经网络的求导
于是，关于θ的损失函数就是Q的负值，为了使 Critic估计的Q最大

然后 Critic 的损失函数就是使用TD差分的方法：
L = 1/N Σ(y_i - Q(s_i, a_i|θ))^2
其中，
- y_i = r + γQ(s' , Actor(s')|θ)
- N: mini-batch的大小
- γ: 折扣因子
'''


import numpy as np
from torch import nn

import rl_utils
from BaseAgent import BaseAgent
from config import *
from rl_utils import ReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, action_dim)
        self.action_bound = action_bound  # action_bound是环境可以接受的动作最大值
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)) * self.action_bound


# 网络的输入是状态和动作拼接后的向量，网络的输出是一个值，表示该状态动作对的价值
class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, *args, **kwargs):
        super(QCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):

        cat = torch.cat([state.float(), action.float()], dim=1)
        x = self.relu(self.fc1(cat))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


class DDPG_Agent(BaseAgent):
    ''' DDPG算法 '''

    def __init__(self, state_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = QCritic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.target_critic = QCritic(state_dim, action_dim).to(device)
        # 初始化目标价值网络并设置和价值网络相同的参数
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化目标策略网络并设置和策略相同的参数
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.sigma = sigma  # 高斯噪声的标准差,均值直接设为0
        self.tau = tau  # 目标网络软更新参数
        self.action_dim = action_dim
        self.device = device
        self.critic_loss = nn.MSELoss()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).item()
        # 给动作添加噪声，增加探索
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        critic_loss = torch.mean(self.critic_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 使用什么链式法则都是自动去进行的
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


if __name__ == "__main__":
    ALG_NAME = 'DDPG'
    print("这是强化学习 " + ALG_NAME + " 算法")
    actor_lr = 3e-4
    critic_lr = 3e-3
    num_episodes = 200
    hidden_dim = 64
    gamma = 0.98
    tau = 0.005  # 软更新参数
    buffer_size = 10000
    minimal_size = 1000
    batch_size = 64
    sigma = 0.01  # 高斯噪声标准差
    DDPG_ENV_ID = "Pendulum-v1"
    env = gym.make(DDPG_ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]

    agent = DDPG_Agent(state_dim, action_dim, action_bound
                       , sigma, actor_lr, critic_lr, tau, gamma, DEVICE)
    ddpg_buffer = ReplayBuffer(buffer_size)
    rl_utils.train_off_policy_agent(env, agent, num_episodes, ddpg_buffer, minimal_size, batch_size, MAX_STEPS,
                                    Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(DDPG_ENV_ID, agent)
