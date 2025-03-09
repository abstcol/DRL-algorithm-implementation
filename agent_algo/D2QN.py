import torch
import gymnasium as gym
import torch.nn as nn
import copy
import random
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from argument import args
import wandb
from agent_algo.utils import MyDeque, Net

# 自动检测是否有 GPU
# 经过测试发现，由于需要不断与环境交互，模型并不能很好利用 GPU，反而会导致训练时长增加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = "cpu"

class Agent:
    """
    DQN 智能体类，用于训练或测试
    """

    def __init__(
            self,
            env: gym.Env,
            mode: str,
            weight_path="none",
    ):
        """
        初始化智能体
        :param env: 环境实例，通过该实例可以方便获得例如状态尺寸等信息
        :param mode: 运行模式 ("train" 或 "test")
        :param weight_path: 仅在测试模式下使用的模型权重路径
        """
        self.env = env
        self.mode = mode
        self.obs_size = self.env.observation_space.shape[0]  # 观测空间维度
        self.action_size = self.env.action_space.n  # 动作空间维度
        self.gamma = args.gamma  # 折扣因子
        self.alpha = args.alpha  # 优先经验回放的 α 参数
        self.epsilon = args.initial_epsilon  # 初始探索率
        self.final_epsilon = args.final_epsilon  # 最小探索率
        self.epsilon_decay = args.decay_epsilon  # 探索率衰减
        self.tau = args.tau  # 目标网络软更新系数
        self.batch_size = args.batch_size  # 训练批量大小

        # 初始化经验回放区
        self.memory = MyDeque(maxlength=args.MAX_EXPERIENCE, obs_size=self.obs_size, batch_size=args.batch_size)
        self.use_prioritized = not args.stop_prio  # 是否使用优先经验回放

        # 初始化 Q 网络和目标网络
        self.main_q = Net(self.obs_size, self.action_size).to(device)  # 主 Q 网络
        if self.mode == "test":
            state_dict = torch.load(weight_path)
            self.main_q.load_state_dict(state_dict)
        self.target_q = Net(self.obs_size, self.action_size).to(device)  # 目标 Q 网络
        self.target_q.load_state_dict(self.main_q.state_dict())  # 目标网络初始化与主网络相同

        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.main_q.parameters(), lr=args.lr)
        if self.use_prioritized:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")  # PER 需要无均值的损失
        else:
            self.criterion = torch.nn.SmoothL1Loss()

        # 训练过程相关参数
        self.id = 0  # 训练步数计数器
        self.training_error = []  # 存储训练误差，便于日志记录

    def get_action(self, obs) -> int:
        """
        根据当前状态选择动作：
        - 训练模式使用 epsilon-greedy 策略
        - 测试模式使用 greedy 策略
        :param obs: 当前观测状态
        :return: 选择的动作
        """
        obs = torch.from_numpy(obs).to(device)

        if self.mode == "train":
            with torch.no_grad():
                # 以 epsilon 的概率随机选择一个动作（探索），否则选择 Q 值最大的动作（利用）
                if np.random.random() < self.epsilon:
                    return self.env.action_space.sample()
            # 选择 Q 网络预测的最佳动作
            action = torch.argmax(self.main_q(obs)).item()
            return action

        # 测试模式下使用贪心策略
        if self.mode == "test":
            action = torch.argmax(self.main_q(obs)).item()
            return action

    def update(self):
        """
        更新 Q 网络
        """
        # 当经验回放池中的数据不足一个批量时，不进行更新
        if len(self.memory) < self.batch_size:
            self.training_error.append(0)
            return

        # 从经验回放池中采样一个批次
        obs, action, reward, next_obs, terminated, w = self.memory.get_batch()

        # 计算当前 Q 值
        q = self.main_q(obs)
        q = q.gather(dim=1, index=action)

        # 计算目标 Q 值
        with torch.no_grad():
            index_q_t = self.main_q(next_obs).argmax(dim=1, keepdims=True)  # 采用 Double DQN 方法
            q_t = self.target_q(next_obs).gather(1, index_q_t)
            q_t = self.gamma * torch.max(q_t, dim=1, keepdim=True).values * (~terminated) + reward
            q_pred = q.squeeze(1)
            td_error = torch.abs(q_t.squeeze(1) - q_pred)  # 计算 TD 误差
            if self.use_prioritized:
                self.memory.update(td_error ** self.alpha)  # 更新优先级

        # 计算损失
        if self.use_prioritized:
            loss = (w * self.criterion(q, q_t)).float().mean()  # 权重调整损失
        else:
            loss = self.criterion(q, q_t).float()
        self.training_error.append(loss.item())

        # 进行梯度优化
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_q.parameters():
            param.grad.data.clamp_(-1, 1)  # 限制梯度范围，防止梯度爆炸
        self.optimizer.step()

        # 软更新目标网络（Polyak 平滑更新）
        self.id += 1
        if self.id % 4 == 0:
            for target_param, local_param in zip(self.target_q.parameters(), self.main_q.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def record(self, obs, action, reward, next_obs, terminated):
        """
        记录经验到经验回放池
        :param obs: 当前状态
        :param action: 采取的动作
        :param reward: 获得的奖励
        :param next_obs: 下一状态
        :param terminated: 结束标志（True 表示回合结束）
        """
        self.memory.append(obs, action, reward, next_obs, terminated)

    def decay_epsilon(self):
        """
        衰减 epsilon（探索率），用于 epsilon-greedy 策略
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def increase_beta(self):
        """
        增加 beta 值（用于优先经验回放，逐渐调整重要性采样权重）
        """
        self.memory.increase_beta()
