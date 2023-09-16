"""
SAC（Soft Actor-Critic）是一种基于最大熵原理的Actor-Critic方法，旨在在强化学习中同时优化奖励和探索。它是适用于连续动作空间的高效学习算法。

其核心思想是在优化期望奖励的同时，也优化策略的熵，从而鼓励探索。这是通过以下目标函数实现的：

J(Q) = E[Q(s, a) - α * log π(a|s)] ，见line.133
J(π) = E[α * log π(a|s) - Q(s, a)] ，见line.171

其中，
- Q(s, a): 代表在状态s下采取动作a的Q值
- π(a|s): 代表在状态s下采取动作a的策略
- α: 一个正的缩放因子，用于平衡奖励和熵

在SAC中，有以下三个重要的更新步骤：
1. Q函数更新: 通过最小化均方TD误差来进行
2. 策略更新: 通过最大化J(π)来进行
3. α更新: 通过最大化J(α) = E[log π(a|s) - α * (log π(a|s) - log π_target)]来进行，其中π_target是一个目标熵

这样，SAC算法能够同时考虑奖励最大化和有效探索，使得它在多种任务中都能实现高效和稳定的学习。

"""




import numpy as np
from torch import nn
from torch.distributions import Normal

import rl_utils
from config import *


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc_mu = nn.Linear(64, action_dim)
        self.fc_std = nn.Linear(64, action_dim)
        self.action_bound = action_bound
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # # 通过均值全连接层计算均值mu
        mu = self.fc_mu(x)
        # 通过标准差全连接层和Soft-plus激活函数计算标准差std
        std = self.softplus(self.fc_std(x))
        # # 根据计算得到的均值和标准差构造正态分布
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # 从构造的正态分布中采样一个动作，rsample()是重参数化采样
        # 计算log π(a|s)
        log_prob = dist.log_prob(normal_sample)
        # 通过tanh函数将动作压缩到[-1, 1]范围内
        action = torch.tanh(normal_sample)
        # 修正动作的对数概率，以考虑tanh函数的影响
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        # 将动作缩放到指定的动作边界内
        action = action * self.action_bound
        return action, log_prob


# 估计 Q(s,a)
class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc_out = torch.nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x, a):
        cat = torch.cat([x.float(), a.float()], dim=1)
        x = self.relu(self.fc1(cat))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''

    def __init__(self, state_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim
                                                   , action_dim).to(
            device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim
                                                   , action_dim).to(
            device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        # 初始化log_alpha变量并使其可学习。log_alpha是SAC算法中的一个重要参数，它影响了探索的强度。
        # 通过优化log_alpha的值，算法可以自适应地调整探索的程度。
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        # 设置目标熵，这是SAC算法中的一个重要参数，用于控制策略的随机性。
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.critic_loss = nn.MSELoss()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        # 后者是熵正则化项，促使策略在选择高奖励的动作时也保持一定的探索性，J(Q) = E[Q(s, a) - α * log π(a|s)]
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        # Q函数既可以估计状态-动作对的价值，同时也考虑策略的熵
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            self.critic_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            self.critic_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络。计算策略损失，它是基于新采样的动作的Q值和熵的。
        new_actions, log_prob = self.actor(states)
        # 不是熵，而是负对数似然
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        # 计算策略损失。SAC算法尝试最小化策略损失，该损失由三部分组成：
        # 1. 熵正则项: -self.log_alpha.exp() * entropy，这部分鼓励策略保持一些探索性
        # 2. Q值项: -torch.min(q1_value, q2_value)，这部分鼓励策略选择有高Q值的动作
        # 请注意，我们是在最小化损失，因此有一个负号，就会最大化当前的函数
        # J(π) = E[α * log π(a|s) - Q(s, a)]
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值。让alpha朝着目标熵靠近
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


if __name__ == "__main__":
    ALG_NAME = 'SAC'
    print("这是强化学习 " + ALG_NAME + " 算法")
    DDPG_ENV_ID = "Pendulum-v1"
    env = gym.make(DDPG_ENV_ID, render_mode=RENDER_MODE_train)
    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 100
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005  # 软更新参数
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = env.action_space.high[0]  # 动作最大值
    # target_entropy = -1
    target_entropy = -env.action_space.shape[0]
    agent = SACContinuous(state_dim, action_dim, action_bound,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, DEVICE)
    buffer = rl_utils.ReplayBuffer(buffer_size)
    rl_utils.train_off_policy_agent(env, agent, num_episodes, buffer, minimal_size, batch_size, MAX_STEPS, Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic_1.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(ENV_ID, agent)
