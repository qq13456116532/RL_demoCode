import random

import gym
import numpy as np
import torch
from torch import nn, optim

import rl_utils
from BaseAgent import BaseAgent
from config import *


class PolicyGradientNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyGradientNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        state = self.relu(self.fc1(state))
        state = self.relu(self.fc2(state))
        action_probs = self.softmax(self.fc3(state))
        return action_probs


class PGAgent(BaseAgent):
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        probs = self.net(state)
        # 网络的输出是一个向量，其中每个元素表示每个可能动作的预测Q值
        # softmax是一个将任何数字向量转换为概率分布的函数。应用softmax后，输出的向量的所有元素的和为1
        # 为了数值稳定性，我们不直接在网络中使用Softmax
        probs = self.net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.cpu().numpy().item()

    def update(self, transition_dict):
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.device)
        dones = torch.BoolTensor(transition_dict['dones']).to(self.device)
        actions_probs = self.net(states)
        taken_action_probs = actions_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        # Compute discounted rewards (return) for each time step
        # 策略梯度方法（尤其是REINFORCE）通常依赖于每一个时间步的回报（或称为未来的累计奖励）来估计策略的性能
        # torch.arange(len(rewards))生成一个从0到len(rewards)-1的整数序列。
        # torch.pow(self.gamma, ...)会对每一个整数i计算\(\gamma^i\)。结果discounts是一个tensor，其值为[1, gamma, gamma^2, gamma^3, ...]。
        discounts = torch.pow(self.reward_gamma, torch.arange(len(rewards)).float().to(self.device))
        # 计算从每个时间步t开始的折扣后的累计奖励
        # rewards[i:]选取从时间步i到最后的奖励
        # discounts[:len(rewards)-i]选取相应的折扣系数
        discounted_rewards = [torch.sum(rewards[i:] * discounts[:len(rewards) - i]) for i in range(len(rewards))]
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        # Calculate the policy gradient loss
        loss = -torch.mean(torch.log(taken_action_probs) * discounted_rewards)
        # Backpropagate the loss and update the network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



    def __init__(self, state_dim, action_dim, device, reward_gamma, learning_rate, epsilon):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reward_gamma = reward_gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.net = PolicyGradientNet(self.state_dim, self.action_dim).to(device=self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()

if __name__ == "__main__":
    ALG_NAME = 'PolicyGradient'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent =  PGAgent(STATE_DIM,ACTION_DIM,device,REWARD_GAMMA,LEARNING_RATE,Epsilon)
    env = gym.make(ENV_ID,render_mode=RENDER_MODE_train)
    rl_utils.train_on_policy_agent(env,agent,TRAIN_EPISODES,Name=ALG_NAME,max_episode_size=MAX_STEPS)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.net.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID, agent)
