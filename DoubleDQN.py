import random

import torch

from BaseAgent import BaseAgent
from DQN import *

epsilon = lambda i: 1 - 0.99 * min(1, i / (TRAIN_EPISODES* MAX_STEPS * 0.1))


class DDQNAgent(BaseAgent):
    #选择动作和DQN没区别
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32,device=self.device)
        # 根据更新次数计算epsilon的值
        eps = epsilon(self.update_count)
        if(random.random()<eps):
            return int(random.random() * self.action_dim)
        else:
            Q = self.qnet(state).cpu().detach().numpy()
            index = np.argmax(Q)
            return index
            # return self.qnet(state).cpu().numpy().argmax(1)[0]


    # transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
    def update(self, transition_dict):
        self.optimizer.zero_grad()
        # 将数据转化为tensor
        states = torch.FloatTensor(transition_dict['states']).to(self.device)
        actions = torch.LongTensor(transition_dict['actions']).to(self.device)
        next_states = torch.FloatTensor(transition_dict['next_states']).to(self.device)
        rewards = torch.FloatTensor(transition_dict['rewards']).to(self.device)
        dones = torch.BoolTensor(transition_dict['dones']).to(self.device)
        # 使用在线网络选择下一个动作
        _, next_actions = self.qnet(next_states).max(1,keepdims = True)
        # 使用目标网络计算下一个状态的Q值
        next_state_values = self.target_qnet(next_states).gather(1, next_actions)
        # 计算目标值
        expected_values = rewards + (self.gamma * next_state_values * (~dones).float())
        # 计算预测的Q值:
        actions = actions.unsqueeze(-1)
        predicted_values = self.qnet(states).gather(1, actions)
        loss = self.loss(predicted_values, expected_values)
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if(self.update_count%10==0):
            rl_utils.sync_networks(self.qnet,self.target_qnet)


    def __init__(self,state_dim,action_dim,lr,gamma,device):
        super(DDQNAgent, self).__init__()
        self.update_count = 0
        self.buffer = rl_utils.ReplayBuffer(BUFFER_SIZE)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.qnet = DQN(state_dim,action_dim).to(device)
        self.target_qnet = DQN(state_dim,action_dim).to(device)
        self.qnet.train()
        self.target_qnet.train()
        rl_utils.sync_networks(self.qnet,self.target_qnet)
        self.optimizer = optim.Adam(self.qnet.parameters(),lr=lr)
        self.loss = nn.MSELoss()
        self.device = device

if __name__ == "__main__":
    ALG_NAME = 'DoubleDQN'
    print("这是强化学习 " + ALG_NAME + " 算法")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddqnAgent = DDQNAgent(STATE_DIM,ACTION_DIM,LEARNING_RATE,REWARD_GAMMA,device)
    env = gym.make(ENV_ID,render_mode=RENDER_MODE_train)
    ddqnAgent = rl_utils.train_off_policy_agent(env,ddqnAgent,TRAIN_EPISODES,ddqnAgent.buffer,128,16,max_episode_size=MAX_STEPS,Name=ALG_NAME)

    folder_path = './Saved_model/' + ALG_NAME
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    torch.save(ddqnAgent.qnet.state_dict(), os.path.join(folder_path, 'model_weights.pth'))
    rl_utils.test_model(env,ddqnAgent)

