"""
策略梯度 PG方法是一类直接在策略空间中进行优化的算法。它们通过增加那些获得高奖励的动作的概率来改善策略。

在 PG算法中，策略是通过参数化函数（如神经网络）来表示的，参数通过梯度上升方法来进行更新。

策略梯度公式可以表示为：
Δθ = α ∇θ log π(a|s;θ) Q(s, a)
其中，
- α: 学习率
- θ: 策略参数
- π(a|s;θ): 当前策略下，在状态s选择动作a的概率
- Q(s, a): 动作a在状态s下的Q值

损失函数可以定义为：
L = -1/N Σ log π(a_i|s_i;θ) Q(s_i, a_i)
其中，
- N: mini-batch的大小

这里，我们使用Q值来评估在状态s下执行动作a的优势。该方法试图最大化预期的累积奖励，通过梯度上升来更新策略参数。
代码中是使用REINFORCE，也就是使用MC方法来估计 Q值，见line.70 ，使用实际奖励来评估
"""

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
        states, actions, rewards, next_states, dones = rl_utils.get_Samples(transition_dict, self.device)
        actions_probs = self.net(states)
        taken_action_probs = actions_probs.gather(1, actions).squeeze(-1)
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

    def __init__(self, state_dim, action_dim, device, reward_gamma, learning_rate):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.reward_gamma = reward_gamma
        self.learning_rate = learning_rate
        self.net = PolicyGradientNet(self.state_dim, self.action_dim).to(device=self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()


if __name__ == "__main__":
    ALG_NAME = 'PolicyGradient'
    print("这是强化学习 " + ALG_NAME + " 算法")
    learning_rate = 1e-3
    num_episodes = 1000
    hidden_dim = 128
    gamma = 0.98
    ENV_ID = 'CartPole-v1'
    env = gym.make(ENV_ID, render_mode=RENDER_MODE_train)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PGAgent(state_dim, action_dim, DEVICE, gamma, learning_rate)
    rl_utils.train_on_policy_agent(env, agent, num_episodes, Name=ALG_NAME, max_episode_size=MAX_STEPS)
    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(agent.net.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(ENV_ID, agent)
