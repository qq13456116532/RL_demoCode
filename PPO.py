import torch
from torch import nn, optim

from BaseAgent import BaseAgent
from config import *


# 策略网络PolicyNet（与 REINFORCE 算法一样）
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, state):
        state = self.relu(self.fc1(state))
        state = self.fc2(state)
        probs = self.softmax(state)
        return probs


# 价值网络ValueNet，其输入是某个状态，输出则是状态的价值
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, state):
        state = self.relu(self.fc1(state))
        value = self.fc2(state)
        return value


class PPOAgent(BaseAgent):
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor_net(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        td_target = rewards + self.gamma * self.critic_net(next_states) * (1-dones)
        td_delta = td_target - self.critic_net(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor_net(states).gather(1,actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor_net(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(self.critic_loss(self.critic_net(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()



    def __init__(self, state_dim, action_dim, reward_gamma, device,lmbda,eps, gamma, epochs,actor_lr=1e-3,crtic_lr=1e-2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_gamma = reward_gamma
        self.lmbda =lmbda
        self.eps = eps  # PPO中截断范围的参数
        self.gamma = gamma
        self.epochs = epochs # 一条序列的数据用来训练轮数
        self.device = device
        self.actor_net = Actor(state_dim, action_dim).to(device)
        self.critic_net = Critic(state_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=crtic_lr)
        self.critic_loss = nn.MSELoss()


if __name__ == "__main__":
    ALG_NAME = 'ppo'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(STATE_DIM, ACTION_DIM, REWARD_GAMMA, device,PPO_lmbda,PPO_eps,REWARD_GAMMA,PPO_epochs,ACTOR_LR,CRITIC_LR)
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    rl_utils.train_on_policy_agent(env, agent, TRAIN_EPISODES, Name=ALG_NAME, max_episode_size=MAX_STEPS)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.actor_net.state_dict(), os.path.join(folder_path, 'model_weights_actor.pth'))
    torch.save(agent.critic_net.state_dict(), os.path.join(folder_path, 'model_weights_critic.pth'))
    rl_utils.test_model(ENV_ID, agent)
