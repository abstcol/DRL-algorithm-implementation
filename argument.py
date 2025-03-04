from typing import List, Union
import argparse

# 创建参数解析器
parser = argparse.ArgumentParser(description="Environment and DQN Hyperparameters")

# 环境相关参数
parser.add_argument(
    "--env-name",
    type=str,
    default="CartPole-v1",
    help="The name of the Gymnasium environment to use (default: CartPole-v1)."
)

parser.add_argument(
    "--episodes",
    type=int,
    default=600,
    help="The total number of episodes to train the agent (default: 600)."
)

parser.add_argument(
    "--max-episode-steps",
    type=int,
    default=400,
    help="The maximum number of steps allowed in a single episode (default: 400)."
)

# 经验回放相关参数
parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="The number of trajectories sampled from the memory buffer for each training step (default: 64)."
)

parser.add_argument(
    "--MAX-EXPERIENCE",
    type=int,
    default=100000,
    help="The maximum size of the experience replay buffer (default: 100000)."
)

# DQN 学习相关参数
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="The discount factor for future rewards (default: 0.99)."
)

parser.add_argument(
    "--tau",
    type=float,
    default=1e-2,
    help="The soft update rate for the target network (default: 1e-2)."
)

parser.add_argument(
    "--alpha",
    type=float,
    default=0.6,
    help="The magnitude of priority sampling in experience replay (default: 0.6)."
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="The learning rate for the DQN optimizer (default: 0.001)."
)

# 探索策略相关参数
parser.add_argument(
    "--initial-epsilon",
    type=float,
    default=0.1,
    help="The initial value of epsilon for the epsilon-greedy exploration strategy (default: 0.1)."
)

parser.add_argument(
    "--decay-epsilon",
    type=float,
    default=0.995,
    help="The decay rate of epsilon after each episode (default: 0.995)."
)

parser.add_argument(
    "--final-epsilon",
    type=float,
    default=0.001,
    help="The minimum value of epsilon for the epsilon-greedy exploration strategy (default: 0.001)."
)

# 神经网络结构相关参数
parser.add_argument(
    "--hidden-sizes",
    type=int,
    nargs="+",
    default=[32, 6],
    help="The sizes of the hidden layers in the DQN (default: [32, 6])."
)

# 解析参数
args = parser.parse_args()