'''
1. Double DQN算法是DQN算法的一个扩展，它通过减少Q值的过估计来提高学习的稳定性和性能。

2. 就像DQN一样，Double DQN也使用一个神经网络来近似Q值函数，旨在学习一个策略来最大化预期奖励。
但不同之处在于，它使用两个网络（在线网络和目标网络）来分别选择动作和评估，以减少过估计偏差。
在线网络-->选择动作line.72      目标网络-->估计下一个状态的Q值line.74

Q值更新公式（Double DQN）：
Δw = α [r + γ Q(s', argmax_a Q(s', a|w)|w^-) - Q(s, a|w)] ∇w Q(s, a|w)  ，看这个公式就知道 Q(s',a')是目标参数评估的，argmax_a是在线网络评估的
其中，
- w: 在线Q网络的参数
- w^-: 目标Q网络的参数

损失函数：
L = 1/N Σ(y_i - Q(s_i, a_i|w))^2
其中，
- y_i = r + γ Q(s' , argmax_a Q(s', a|w)|w^-)
- N: mini-batch的大小
- γ: 折扣因子

通过引入一个额外的Q网络来选择动作（而不是评估它们），Double DQN有助于减少Q值的过估计，从而提供更稳定和更高效的训练。
'''


import numpy as np
import torch.nn.functional as F
from torch import nn

import rl_utils
from BaseAgent import BaseAgent
from config import *


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()  # 这一行是关键
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.state_dim, 128)
        self.RELU = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, self.action_dim)

    def forward(self, input):
        # input = torch.Tensor(input)
        input = self.RELU(self.fc1(input))
        input = self.RELU(self.fc2(input))
        # 注意这里不应使用ReLU
        output = self.fc3(input)
        # 预测每个动作的Q值
        return output


class DDQNAgent(BaseAgent):
    # 选择动作和DQN没区别
    def take_action(self, state):
        if (np.random.random() < self.epsilon):
            # 小概率完全随机
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            Q = self.qnet(state).cpu().detach().numpy()
            index = np.argmax(Q)
            return index

    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)

        q_values = self.qnet(states).gather(1, actions)  # Q值
        max_action = self.qnet(next_states)
        max_action = max_action.max(1)[1].view(-1, 1)
        next_q_values = self.target_qnet(next_states)
        max_next_q_values = next_q_values.gather(1, max_action)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update == 0:
            rl_utils.sync_networks(self.qnet, self.target_qnet)

    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        super(DDQNAgent, self).__init__()
        self.update_count = 1
        self.action_dim = action_dim
        self.qnet = DQN(state_dim, action_dim).to(device)
        self.target_qnet = DQN(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device


if __name__ == "__main__":
    ALG_NAME = 'DoubleDQN'
    lr = 1e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.98
    # ε-greedy策略
    epsilon = 0.01
    # 目标网络什么时候load
    target_update = 10
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    print("这是强化学习 " + ALG_NAME + " 算法")
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    buffer = rl_utils.ReplayBuffer(buffer_size)
    ddqnAgent = DDQNAgent(state_dim, action_dim, lr, gamma, epsilon,
                          target_update, DEVICE)
    rl_utils.train_off_policy_agent(env, ddqnAgent, num_episodes, buffer, minimal_size, batch_size,
                                    max_episode_size=MAX_STEPS, Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(ddqnAgent.qnet.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(env, ddqnAgent)
