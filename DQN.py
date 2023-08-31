import os
import time

import numpy as np
import torch
import gym
from tensorboardX import SummaryWriter
from torch import nn, optim
from tqdm import tqdm

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
        if(np.random.random()<Epsilon):
            #小概率完全随机
            return np.random.choice(self.action_dim)
        else:
            state = torch.FloatTensor(state).to(self.device)
            # state = torch.FloatTensor(state)
            Q   = self.dqn_net(state).cpu().detach().numpy()
            index = np.argmax(Q)
            return index


    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        # 将数据转化为tensor
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.device)
        dones = torch.BoolTensor(transition_dict['dones']).to(self.device)
        # 清空梯度
        self.optimizer.zero_grad()
        # 当前状态下的Q值
        Q_values = self.dqn_net(states)
        Q = Q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 下一个状态的Q值
        next_Q_values = self.dqn_net(next_states)
        max_next_Q, _ = torch.max(next_Q_values, dim=1)
        # 计算目标Q值
        targetQ = rewards + (1 - dones.float()) * self.reward_gamma * max_next_Q
        # 计算TD误差
        td_error = self.loss(Q, targetQ)
        td_error.backward()
        self.optimizer.step()



    def __init__(self,state_dim,action_dim,learning_rate,reward_gamma,device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.device = device
        self.dqn_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.dqn_net.train()
        self.buffer = rl_utils.ReplayBuffer(BUFFER_SIZE)
        self.loss = nn.HuberLoss()
        self.optimizer = optim.Adam(self.dqn_net.parameters(),lr=learning_rate, eps=1e-8)


if __name__ == "__main__":
    ALG_NAME = 'DQN'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqnAgent = DQNAgent(STATE_DIM,ACTION_DIM,LEARNING_RATE,REWARD_GAMMA,device)
    env = gym.make(ENV_ID,render_mode=RENDER_MODE_train)
    dqnAgent = rl_utils.train_off_policy_agent(env,dqnAgent,TRAIN_EPISODES,dqnAgent.buffer,128,16,max_episode_size=MAX_STEPS,Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(dqnAgent.qnet.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID,dqnAgent)




