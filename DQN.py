'''
DQN（Deep Q-Network）算法是一个用深度学习优化的Q学习算法，它主要用于离散动作空间。

DQN算法基本上是Q-learning的一个扩展，使用神经网络来近似Q函数，而不是使用传统的表格方法，
还引入了经验回放和目标网络来稳定训练。


Q值更新公式（DQN）：
Δw = α [r + γ max Q(s', a'|w^-) - Q(s, a|w)] ∇w Q(s, a|w)
其中，
- w: Q网络的参数
- w^-: 目标Q网络的参数


损失函数就是使用TD差分的方式：
L = 1/N Σ(y_i - Q(s_i, a_i|w))^2
其中，
- y_i = r + γ max Q(s' , a'|w^-)
- N: mini-batch的大小
- γ: 折扣因子

经验回放机制和目标网络是DQN的两个关键组件，有助于避免训练中的不稳定和发散。
'''


import numpy as np
from torch import nn, optim
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


class DQNAgent(BaseAgent):
    def take_action(self, state):
        if (np.random.random() < self.epsilon):
            # 小概率完全随机
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            # state = torch.FloatTensor(state)
            Q = self.dqn_net(state).cpu().detach().numpy()
            index = np.argmax(Q)
            return index

    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict,self.device)
        # 清空梯度
        self.optimizer.zero_grad()
        # 当前状态下的Q值
        Q_values = self.dqn_net(states)
        Q = Q_values.gather(1, actions)
        # 下一个状态的Q值
        next_Q_values = self.dqn_net(next_states)
        max_next_Q, _ = torch.max(next_Q_values, dim=1)
        # 计算目标Q值
        targetQ = rewards + (1 - dones) * self.reward_gamma * max_next_Q.unsqueeze(1)
        # 计算TD误差
        td_error = self.loss(Q, targetQ)
        td_error.backward()
        self.optimizer.step()

    def __init__(self, state_dim, action_dim, learning_rate, reward_gamma, buffer_size, device, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.device = device
        self.dqn_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.dqn_net.train()
        self.buffer = rl_utils.ReplayBuffer(buffer_size)
        self.epsilon = epsilon
        self.loss = nn.HuberLoss()
        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=learning_rate, eps=epsilon)


if __name__ == "__main__":
    ALG_NAME = 'DQN'
    DQN_learning_rate = 2e-3
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98
    # ε-greedy策略
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    # ==================================以上是超参数设置==============================================
    print("这是强化学习 " + ALG_NAME + " 算法")
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dqnAgent = DQNAgent(state_dim, action_dim, DQN_learning_rate, gamma, buffer_size, DEVICE, epsilon)
    rl_utils.train_off_policy_agent(env, dqnAgent, num_episodes, dqnAgent.buffer, minimal_size, batch_size,
                                    max_episode_size=MAX_STEPS, Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(dqnAgent.dqn_net.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID, dqnAgent)
