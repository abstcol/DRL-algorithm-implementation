from typing import List, Union
import argparse

def check_pth_file(value):
    """
    检查给定的路径是否为 .pth 文件。
    :param value: 输入的文件路径
    :return: 如果是 .pth 文件，返回路径，否则抛出错误
    """
    if not value.endswith(".pth"):
        raise argparse.ArgumentTypeError("The weight path must be a .pth file.")
    return value

# 创建参数解析器
parser = argparse.ArgumentParser(description="Environment and DQN Hyperparameters")

# 环境相关参数
parser.add_argument(
    "--env-name",
    type=str,
    default="MountainCar-v0",
    help="The name of the Gymnasium environment to use (default: MountainCar-v0)."
)

parser.add_argument(
    "--episodes",
    type=int,
    default=1000,
    help="The total number of episodes to train the agent (default: 1000)."
)

parser.add_argument(
    "--max-episode-steps",
    type=int,
    default=0,
    help="The maximum number of steps allowed in a single episode (default: 0)."
)

# 使用的算法
parser.add_argument(
    "--algo-name",
    type=str,
    default="D3QN",
    help="The name of the algorithm to use (default: D3QN)."
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
    default=2**17,
    help="The maximum size of the experience replay buffer (default: 2^17)."
)

# DQN 学习相关参数
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    help="The discount factor for future rewards (default: 0.9)."
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
    "--initial-beta",
    type=float,
    default=0.4,
    help="The magnitude of importance sampling in experience replay (default: 0.4)."
)

parser.add_argument(
    "--increase-beta",
    type=float,
    default=1.0,
    help="The increase rate of the magnitude of importance sampling in experience replay (default: 1.0)."
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
    default=0.99,
    help="The decay rate of epsilon after each episode (default: 0.99)."
)

parser.add_argument(
    "--final-epsilon",
    type=float,
    default=0.01,
    help="The minimum value of epsilon for the epsilon-greedy exploration strategy (default: 0.01)."
)

parser.add_argument(
    "--prioepsilon",
    type=float,
    default=1e-6,
    help="The epsilon for the prioritized sampling strategy (default: 1e-6)."
)

parser.add_argument(
    "--stop-prio",
    action="store_true",
    help="If set, the network won't use prioritized sampling."
)

# 神经网络结构相关参数
parser.add_argument(
    "--hidden-sizes",
    type=int,
    nargs="+",
    default=[64, 32],
    help="The sizes of the hidden layers in the DQN (default: [64, 32])."
)

# # 测试相关参数
# parser.add_argument(
#     "--weight-path",
#     type=check_pth_file,
#     default=".pth",
#     help="The weight path used in test."
# )

# 生成 gif
parser.add_argument(
    "--produce-gif",
    action="store_true",
    help="Produce gif, gif saved to trun_rl/gifs/{env_name}_{algo_name}_{time_stamp}.gif."
)

parser.add_argument(
    "--no-human-sample",
    action="store_true",
    help="If set, no human samples will be used."
)

# wandb 权重的名称
parser.add_argument(
    "--checkpoint-name",
    type=str,
    default='abstcold-none/RL-Training/MountainCar-v0_D3QN_20250308-204502:v18',
    help="The checkpoint name you can get on wandb."
)

# 解析参数
args = parser.parse_args()
