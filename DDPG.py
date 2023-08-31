# 适用于连续动作空间
import numpy as np
import torch
from torch import nn

from BaseAgent import BaseAgent
from config import *


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
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)
        x = self.relu(self.fc1(cat))
        x = self.relu(self.fc2(x))
        return self.fc_out(x)

class DDPG_Agent(BaseAgent):
    ''' DDPG算法 '''
    def __init__(self, state_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        super().__init__()
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = QCritic(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim,  action_dim, action_bound).to(device)
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
        self.critic_loss =nn.MSELoss()

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

        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)  # 软更新策略网络
        self.soft_update(self.critic, self.target_critic)  # 软更新价值网络


if __name__ == "__main__":
    ALG_NAME = 'DDPG'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(DDPG_ENV_ID, render_mode=RENDER_MODE_train)

    agent = DDPG_Agent(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high[0] , DDPG_sigma,DDPG_actor_lr,DDPG_critic_lr,DPPG_tau,DDPG_gamma,device)
    rl_utils.train_off_policy_agent(env, agent, TRAIN_EPISODES, buffer, 128,16,MAX_STEPS,Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(DDPG_ENV_ID, agent)