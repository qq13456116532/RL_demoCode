'''
引入一个双分支神经网络来分别估计状态值函数V(s)和优势函数A(s, a)，从而提高Q值的估计精度和学习稳定性。

该算法的网络结构包含两个部分:
1. 值网络(Value Network): 估计某一状态的值，而不考虑采取特定动作的影响。
2. 优势网络(Advantage Network): 估计采取每一动作相对于平均动作的优势或劣势。

Q值由值网络和优势网络共同决定，计算公式如下：
Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))

损失函数仍然基于TD误差，形式如下：
L = 1/N Σ(y_i - Q(s_i, a_i|w))^2
其中，
- y_i = r + γ max Q(s', a'|w)
- N: mini-batch的大小
- γ: 折扣因子

Dueling DQN算法旨在更好地区分状态值和动作优势，从而使学习更加高效和稳定。
'''

import random
from torch import nn, optim
import rl_utils
from BaseAgent import BaseAgent
from config import *


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        # 每个动作的优势
        self.fcA = nn.Linear(64, action_dim)
        # 状态的V(s)
        self.fcV = nn.Linear(64, 1)

    def forward(self, state):
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        advantages = self.fcA(state)
        value = self.fcV(state)
        # return advantages,value
        # Q-values are calculated as per the Dueling DQN approach:
        # 双重 DQN 结构中，我们不是单独返回 advantages 和 value，而是返回 Q-values，这是由 value 和 advantages 结合而来的
        qvals = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals


class DuelingAgent(BaseAgent):
    def take_action(self, state):
        if (random.random() < self.epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.dueling_dqn(state)
            return q_values.argmax().item()

    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)
        q_values = self.dueling_dqn(states)
        q_values = q_values.gather(1, actions.long())
        max_next_q_values = self.dueling_dqn(next_states)
        max_next_q_values, _ = max_next_q_values.max(dim=1, keepdim=True)
        q_tragets = self.gamma * max_next_q_values * (1-dones) + rewards
        dqn_loss = torch.mean(self.loss(q_values, q_tragets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

    def __init__(self, device, state_dim, action_dim, gamma, epsilon):
        super().__init__()
        self.device = device
        self.dueling_dqn = DuelingDQN(state_dim, action_dim).to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.dueling_dqn.parameters())
        self.loss = nn.MSELoss()
        self.gamma = gamma
        self.epsilon = epsilon


if __name__ == "__main__":
    ALG_NAME = 'DuelingDQN'
    print("这是强化学习 " + ALG_NAME + " 算法")
    lr = 1e-3
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.98
    # ε-greedy策略
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    dueling_dqn = DuelingAgent(DEVICE, state_dim, action_dim, gamma,epsilon)
    buffer = rl_utils.ReplayBuffer(buffer_size)
    rl_utils.train_off_policy_agent(env, dueling_dqn, num_episodes, buffer, minimal_size, batch_size,
                                                max_episode_size=MAX_STEPS, Name=ALG_NAME)

    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(dueling_dqn.dueling_dqn.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID, dueling_dqn)
