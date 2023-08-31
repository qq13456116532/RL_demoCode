# 适用于连续动作空间,DDPG的改进
import numpy as np
import torch
from torch import nn, optim

import rl_utils
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


"""
     这个类实现了TD3（双延迟深度确定性策略梯度）算法。
     它从BaseAgent类继承。
 """


class TD3_Agent(BaseAgent):
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
        actor_actions = self.actor(states)
        # 使用value网络评估动作的价值,价值是越高越好
        qvalue1 = self.critic1(states,actor_actions)
        qvalue2 = self.critic2(states,actor_actions)
        actor_loss = -torch.min(qvalue1, qvalue2).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ===========================上面是更新actor======================================
        value1 = self.critic1(states,actions)
        value2 = self.critic2(states,actions)
        # 计算target
        next_actions = self.actor_delay(next_states)
        with torch.no_grad():
            target1 = self.critic1_delay(next_states,next_actions)
            target2 = self.critic2_delay(next_states,next_actions)
        target = torch.min(target1, target2)
        target = target * 0.99 * (1 - dones) + rewards
        critic1_loss = nn.MSELoss()(target,value1)
        critic2_loss = nn.MSELoss()(target,value2)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # ===========================上面是更新两个critic======================================
        self.soft_update(self.critic1,self.critic1_delay)
        self.soft_update(self.critic2,self.critic2_delay)
        self.soft_update(self.actor,self.actor_delay)





    """
            使用给定的参数初始化TD3代理。

            :param state_dim: 状态空间的维度。
            :param action_dim: 动作空间的维度。
            :param action_bound: 动作值的边界。
            :param sigma: 用于探索的噪声的标准差。
            :param actor_lr: actor网络的学习率。
            :param critic_lr: critic网络的学习率。
            :param tau: 软更新目标参数的因子。
            :param gamma: 未来奖励的折扣因子。
            :param device: 模型应分配的设备（例如，'cuda'表示GPU，'cpu'表示CPU）。
    """
    def __init__(self, state_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device):
        super(TD3_Agent, self).__init__()
        # 初始化主actor及其延迟（目标）副本
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        # 延迟的目标网络,提高算法的稳定性
        self.actor_delay = Actor(state_dim, action_dim, action_bound).to(device)
        self.actor_delay.load_state_dict(self.actor.state_dict())
        # 初始化第一个critic及其延迟（目标）副本
        # 两个Critic网络以减少值估计的过高估计。当决策是基于过高估计的值进行时，可能导致性能下降。
        self.critic1 = QCritic(state_dim, action_dim).to(device)
        self.critic1_delay = QCritic(state_dim, action_dim).to(device)
        self.critic1_delay.load_state_dict(self.critic1.state_dict())
        # 初始化第二个critic及其延迟（目标）副本
        self.critic2 = QCritic(state_dim, action_dim).to(device)
        self.critic2_delay = QCritic(state_dim, action_dim).to(device)
        self.critic2_delay.load_state_dict(self.critic2.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(),lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(),lr=critic_lr)

        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim


if __name__ == "__main__":
    ALG_NAME = 'TD3'
    print("这是强化学习 " + ALG_NAME + " 算法")
    env = gym.make(DDPG_ENV_ID, render_mode=RENDER_MODE_train)
    agent = TD3_Agent(env.observation_space.shape[0], env.action_space.shape[0],env.action_space.high[0] , DDPG_sigma,DDPG_actor_lr,DDPG_critic_lr,DPPG_tau,DDPG_gamma,DEVICE)
    rl_utils.train_off_policy_agent(env, agent, TD3_TRAIN_EPISODES, buffer, 128, 16, MAX_STEPS, Name=ALG_NAME)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    # torch.save(agent.critic1.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(DDPG_ENV_ID, agent)



