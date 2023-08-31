import random

import torch
from torch import nn, Tensor, optim

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
        if (random.random() < Epsilon):
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.dueling_dqn(state)
            return q_values.argmax().item()

    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.device)
        dones = torch.BoolTensor(transition_dict['dones']).to(self.device)
        q_values = self.dueling_dqn(states)
        q_values = q_values.gather(1, actions.unsqueeze(-1))
        max_next_q_values = self.dueling_dqn(next_states)
        max_next_q_values, _ = max_next_q_values.max(dim=1, keepdim=True)
        q_tragets = self.gamma * max_next_q_values * (~ dones).float() + rewards
        dqn_loss = torch.mean(self.loss(q_values, q_tragets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

    def __init__(self, device, state_dim, action_dim, gamma):
        self.device = device
        self.dueling_dqn = DuelingDQN(state_dim, action_dim).to(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.optimizer = optim.Adam(self.dueling_dqn.parameters())
        self.loss = nn.MSELoss()
        self.gamma = gamma


if __name__ == "__main__":
    ALG_NAME = 'DuelingDQN'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dueling_dqn = DuelingAgent(device, STATE_DIM, ACTION_DIM, REWARD_GAMMA)
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    rl_utils.train_off_policy_agent(env, dueling_dqn, TRAIN_EPISODES, buffer, 128, 16,
                                                max_episode_size=MAX_STEPS, Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(dueling_dqn.dueling_dqn.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID, dueling_dqn)
