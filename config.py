import os
import gym
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


RENDER_MODE_train = "rgb_array"
RENDER_MODE_test = "human"

# 测试总次数，暂时还没用，可以在rl_utils.py的test_model()中使用
TEST_EPISODES = 10

MINIMAL_SIZE = 128
BATCH_SIZE = 16

# PG needs 500 ，DQNs just 50
MAX_STEPS = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
