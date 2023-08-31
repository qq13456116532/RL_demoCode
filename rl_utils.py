import gym
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

import collections
import random
import numpy as np

import BaseAgent
from tensorboardX import SummaryWriter


class ReplayBuffer:
    def __init__(self, capacity):
        # 初始化一个双端队列，它的最大长度为capacity
        # 自动处理溢出，确保缓冲区始终包含最近的capacity个经验，而不需要手动删除旧数据
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # 向缓冲区添加一个经验元组：(状态, 动作, 奖励, 下一个状态, 完成标志)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 从缓冲区随机采样一个批次的经验
        # transitions实际上是一个五元组的列表
        transitions = random.sample(self.buffer, batch_size)

        # 解压经验批次，得到五个列表：状态、动作、奖励、下一个状态和完成标志
        # zip函数解压缩transitions中的经验
        state, action, reward, next_state, done = zip(*transitions)

        # 返回numpy数组形式的经验批次
        return (
            np.array(state),
            np.array(action),
            np.array(reward),
            np.array(next_state),
            np.array(done, dtype=np.bool_)
        )

    def size(self):
        # 返回缓冲区中的经验数量
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)

# transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
def get_Samples(transition_dict,device):
    states = torch.tensor(transition_dict['states'], dtype=torch.float).to(device)
    actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1, 1).to(device)
    rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(device)
    next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(device)
    dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(device)
    return states,actions,rewards,next_states,dones

def moving_average(a, window_size):
    """
    计算给定数组的移动平均值。

    :param a: 输入的数组。
    :param window_size: 移动窗口的大小。
    :return: 一个新的数组，表示输入数组的移动平均值。
    """
    # 计算累计和，注意在数组的开始位置添加了一个0，是为了后续计算方便
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    # 中间部分的移动平均值是通过取累计和的差，然后除以窗口大小得到的。
    # 例如：cumulative_sum[i+window_size] - cumulative_sum[i] 给出了i到i+window_size之间的和
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    # 创建一个范围数组，用于计算起始部分的平均值
    r = np.arange(1, window_size - 1, 2)
    # 计算起始部分的移动平均值。
    # 使用cumsum计算累计和，然后每两个元素取一个，最后除以r以得到平均值
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    # 计算结尾部分的移动平均值，这部分和计算起始部分的方法类似，但是是从数组的末尾开始计算的
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]

    # 将起始、中间和结尾部分的移动平均值连接成一个完整的数组，并返回
    return np.concatenate((begin, middle, end))


# 导入tqdm库来显示进度条
from tqdm import tqdm
import numpy as np


# 定义一个函数，该函数在给定环境中训练on-policy代理。
def train_on_policy_agent(env, agent: BaseAgent, num_episodes, Name,max_episode_size):
    # 初始化一个列表用于保存每一个情节的总回报
    return_list = []
    writer = SummaryWriter('./tensorboard/exp_' + Name)
    # 主循环运行10次，每次训练num_episodes/10个情节
    for i in range(10):
        # 使用tqdm来显示进度条，方便观察训练过程
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            # 内部循环用于实际的情节训练
            for i_episode in range(int(num_episodes / 10)):
                # 初始化情节的回报为0
                episode_return = 0
                # 初始化一个字典来保存转换（状态，动作，奖励等）
                # 上次update之后就进行清空，不使用replaybuffer
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                # 重置环境并获取初始状态
                state, info = env.reset()
                done = False
                # 持续采取动作直到情节结束
                while not done:
                    # 代理决定在当前状态下要采取的动作
                    action = agent.take_action(state)
                    # 在环境中采取动作并获得下一个状态，奖励，是否结束标志等
                    next_state, reward, done, t, _ = env.step(action)
                    # 保存这些转换信息
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    # 将当前状态更新为下一个状态
                    state = next_state
                    # 更新情节的总回报
                    episode_return += reward
                    if len(transition_dict['dones'])>max_episode_size:
                        break ;
                writer.add_scalar('episode_return', episode_return, global_step=i_episode + i * int(num_episodes / 10))
                # 保存该情节的总回报
                return_list.append(episode_return)
                # 使用保存的转换信息更新代理
                agent.update(transition_dict)
                # 每10个情节更新一次进度条的信息
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                # 更新进度条
                pbar.update(1)
    writer.close()
    # 返回所有情节的总回报列表
    return return_list


def train_off_policy_agent(env, agent: BaseAgent, num_episodes, replay_buffer: ReplayBuffer, minimal_size, batch_size,
                           max_episode_size, Name):
    """
    训练一个off-policy智能体

    :param env: 环境对象，必须提供reset和step等接口
    :param agent: 智能体对象，需要提供take_action和update接口
    :param num_episodes: 需要训练的总的回合数
    :param replay_buffer: 经验回放缓冲区
    :param minimal_size: 在开始学习前，回放缓冲区中的最小经验数量
    :param batch_size: 从回放缓冲区中一次采样的经验数量
    :param max_episode_size 一次episode最大次数
    :param Name 设置TensorBoard的名字

    :return: 返回每个回合的总回报列表
    """

    return_list = []  # 用于存储每个回合的总回报
    writer = SummaryWriter('./tensorboard/exp_' + Name)
    # 外层循环是为了利用tqdm进度条，将训练分为10个子迭代
    for iter_num in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % iter_num) as pbar:
            # 内层循环对每个回合进行迭代
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0  # 当前回合的总回报
                episode_count = 0
                state, info = env.reset()  # 重置环境获取初始状态
                done = False  # 回合是否结束

                # 回合内部循环
                while not done:
                    action = agent.take_action(state)  # 智能体根据当前状态选择动作
                    next_state, reward, done, t, _ = env.step(action)  # 在环境中执行动作
                    episode_count += 1
                    replay_buffer.add(state, action, reward, next_state, done)  # 将经验添加到回放缓冲区
                    state = next_state  # 更新状态
                    episode_return += reward  # 累计回报
                    if (episode_count > max_episode_size):
                        break;

                    # 当回放缓冲区的大小大于预定的最小值时，开始训练智能体
                    if replay_buffer.size() > minimal_size:
                        # 从回放缓冲区中随机采样
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)  # 更新智能体的策略
                writer.add_scalar('episode_return', episode_return,
                                  global_step=i_episode + iter_num * int(num_episodes / 10))
                return_list.append(episode_return)  # 将当前回合的总回报添加到列表中

                # 每10个回合更新一次进度条信息
                #  # 每10个回合更新一次进度条的后缀信息，显示最近10个回合的平均回报
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * iter_num + i_episode + 1),
                                      '最新十次的return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)  # 更新进度条

    writer.close()
    # return return_list  # 返回总回报列表
    # 保存模型权重
    return agent


# Generalized Advantage Estimation (GAE)。GAE 是一种为优势函数估计提供更平滑估计的技术
# 可以先去了解一下GAE公式
def compute_advantage(gamma, lmbda, td_delta):
    """
    使用 GAE 方法计算优势值。

    参数:
    gamma (float): 折扣因子，通常在 [0, 1] 之间。
    lmbda (float): GAE 衰减因子，通常在 [0, 1] 之间。
    td_delta (torch.Tensor): TD(时间差分)误差，表示价值函数的估计误差。

    返回:
    torch.Tensor: 计算得到的优势值列表。
    """

    # 将 td_delta 转为 numpy 数组，因为稍后我们需要进行迭代计算
    td_delta = td_delta.detach().numpy()

    # 初始化一个空的优势值列表
    advantage_list = []

    # 设置初始优势为 0.0
    advantage = 0.0

    # 从后往前迭代计算优势值
    # 这里使用了[::-1]对 td_delta 进行逆序处理
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)

    # 因为我们是从后往前计算，所以最后要将结果进行逆序操作
    advantage_list.reverse()

    # 返回优势值列表，转化为 torch.Tensor 类型
    return torch.tensor(advantage_list, dtype=torch.float)


def sync_networks(source_net, target_net):
    """
    将source_net的权重复制到target_net.
    """
    target_net.load_state_dict(source_net.state_dict())


def test_model(env_id, agent: BaseAgent):
    env = gym.make(env_id, render_mode="human")
    state, info = env.reset()
    count = 0
    while True:
        action = agent.take_action(state)
        state, reward, done, t, _ = env.step(action)
        if done:
            print("the return value of this test is : " + str(count))
            # print("if you want to continue ,please enter C else E : ")
            state ,info = env.reset()
            count = 0
            # user_in = input()
            # if user_in.upper() == "C":
            #     continue
        count = count + reward


