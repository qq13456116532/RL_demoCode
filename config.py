import os
import gym
import torch

import rl_utils

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

ENV_ID = 'CartPole-v1'
RENDER_MODE_train = "rgb_array"
RENDER_MODE_test = "human"
# PG needs 10000，DQNs just 1000
TRAIN_EPISODES = 1000
TD3_TRAIN_EPISODES = 200
TEST_EPISODES = 10
# PG needs 500 ，DQNs just 50
MAX_STEPS = 500
LAM = 0.9
LEARNING_RATE = 2e-4
BUFFER_SIZE = 1024
# ε-greedy策略
Epsilon = 0.1
REWARD_GAMMA = 0.99
PPO_lmbda = 0.95
PPO_eps = 0.2
PPO_epochs = 10
ACTOR_LR = 1e-3
CRITIC_LR = 1e-2
DDPG_actor_lr = 3e-4
DDPG_critic_lr = 3e-3
DDPG_gamma = 0.98
DPPG_tau = 0.005  # 软更新参数
DDPG_sigma = 0.01  # 高斯噪声标准差
DDPG_ENV_ID = "Pendulum-v1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env_train = gym.make(ENV_ID, render_mode=RENDER_MODE_train)

STATE_DIM = env_train.observation_space.shape[0]
ACTION_DIM = env_train.action_space.n

buffer = rl_utils.ReplayBuffer(BUFFER_SIZE)
