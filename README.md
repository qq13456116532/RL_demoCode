# RL_demoCode

A comprehensive collection of Reinforcement Learning algorithm examples, designed for both research and educational purposes.

## Prerequisites

- **gym**: latest version
- **python**: 3.9

## Usage

For any algorithm example you choose to run, like `DQN.py`, execution will result in the creation of both a TensorBoard file and a model file. To visualize the progress and performance of the Reinforcement Learning algorithm:

1. Execute your desired RL algorithm:

   ```python
   python DQN.py
   ```

2. Launch TensorBoard using the generated log file:

   ```python
   tensorboard --logdir=[path_to_your_tensorboard_file] --port=6006
   ```

This allows you to track and visualize the algorithm's improvement over time.
