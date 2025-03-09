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
from agent_algo.utils import *

# 设备选择，默认使用 CPU
# 由于 DQN 需要频繁与环境交互，GPU 并不能显著加速训练，反而会增加训练时长
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Agent:
    """
    DQN 智能体类，用于训练或测试
    """

    def __init__(self, env: gym.Env, mode: str, weight_path="none"):
        """
        初始化智能体

        :param env: 环境实例，包含状态空间、动作空间等信息
        :param mode: 训练模式 ("train" or "test")
        :param weight_path: 仅在测试模式下使用，指定加载的模型权重
        """
        self.env = env
        self.mode = mode
        self.obs_size = self.env.observation_space.shape[0]  # 观察空间维度
        self.action_size = self.env.action_space.n  # 动作空间大小

        # DQN 训练超参数
        self.gamma = args.gamma  # 折扣因子
        self.alpha = args.alpha  # 经验回放的优先级调整参数
        self.epsilon = args.initial_epsilon  # 初始探索率
        self.final_epsilon = args.final_epsilon  # 最小探索率
        self.epsilon_decay = args.decay_epsilon  # 探索率衰减系数
        self.tau = args.tau  # 目标网络软更新参数
        self.batch_size = args.batch_size  # 训练批量大小

        # 经验回放缓冲区（Replay Buffer）
        self.memory = MyDeque(
            maxlength=args.MAX_EXPERIENCE,  # 经验回放区最大容量
            obs_size=self.obs_size,
            batch_size=args.batch_size
        )
        self.use_prioritized = not args.stop_prio  # 是否使用优先经验回放（Prioritized Experience Replay）

        # 初始化 Q 网络（Main Q 网络 & Target Q 网络）
        self.main_q = Net(self.obs_size, self.action_size).to(device)  # 训练用的主 Q 网络

        if self.mode == "test":
            state_dict = torch.load(weight_path)  # 加载测试模型权重
            self.main_q.load_state_dict(state_dict)

        self.target_q = Net(self.obs_size, self.action_size).to(device)  # 目标 Q 网络
        self.target_q.load_state_dict(self.main_q.state_dict())  # 初始化目标网络参数

        # 优化器 & 损失函数
        self.optimizer = torch.optim.Adam(self.main_q.parameters(), lr=args.lr)
        if self.use_prioritized:
            self.criterion = torch.nn.SmoothL1Loss(reduction="none")  # Huber 损失（用于优先级经验回放）
        else:
            self.criterion = torch.nn.SmoothL1Loss()  # Huber 损失（标准 DQN）

        # 训练相关
        self.id = 0  # 训练步数计数器
        self.training_error = []  # 记录损失值（用于日志记录）

    def get_action(self, obs) -> int:
        """
        根据当前状态选择动作：
        - 训练模式使用 epsilon-greedy 策略（探索+利用）
        - 测试模式使用 greedy 策略（纯利用）

        :param obs: 当前环境状态
        :return: 选定的动作
        """
        obs = torch.from_numpy(obs).to(device)

        if self.mode == "train":
            with torch.no_grad():
                # 以 epsilon 概率随机选择一个动作（探索）
                if np.random.random() < self.epsilon:
                    return self.env.action_space.sample()  # 随机动作（探索）

            # 否则选择 Q 值最大的动作（利用）
            action = torch.argmax(self.main_q(obs)).item()
            return action

        if self.mode == "test":
            action = torch.argmax(self.main_q(obs)).item()  # 仅选择最优动作
            return action

    def update(self):
        """
        使用经验回放进行 Q 网络更新
        """
        # 确保缓冲区中的经验足够一个 batch，否则跳过更新
        if len(self.memory) < self.batch_size:
            self.training_error.append(0)
            return

        # 从经验回放缓冲区采样一个批次
        obs, action, reward, next_obs, terminated, w = self.memory.get_batch()

        # 计算当前 Q 值
        q = self.main_q(obs).gather(dim=1, index=action)  # 只选择执行的动作对应的 Q 值

        # 计算目标 Q 值
        with torch.no_grad():
            index_q_t = self.main_q(next_obs).argmax(dim=1, keepdims=True)  # 选择下一个状态的最大 Q 值动作
            q_t = self.target_q(next_obs).gather(1, index_q_t)  # 从目标 Q 网络获取 Q 值
            q_t = self.gamma * torch.max(q_t, dim=1, keepdim=True).values * (~terminated) + reward

            # 计算时序差分误差（TD Error）
            q_pred = q.squeeze(1)
            td_error = torch.abs(q_t.squeeze(1) - q_pred)  # 计算 TD 误差（用于优先经验回放）

            if self.use_prioritized:
                self.memory.update(td_error**self.alpha)  # 更新优先级

        # 计算损失（Huber Loss）
        if self.use_prioritized:
            loss = (w * self.criterion(q, q_t)).float().mean()  # 考虑样本权重
        else:
            loss = self.criterion(q, q_t).float()  # 标准 DQN 损失
        self.training_error.append(loss.item())

        # 反向传播 + 梯度裁剪
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_q.parameters():
            param.grad.data.clamp_(-1, 1)  # 限制梯度大小，防止梯度爆炸
        self.optimizer.step()

        # 软更新目标 Q 网络
        self.id += 1
        if self.id % 4 == 0:  # 每 4 轮更新一次目标网络
            for target_param, local_param in zip(self.target_q.parameters(), self.main_q.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def record(self, obs, action, reward, next_obs, terminated):
        """
        将经验存储到经验回放缓冲区

        :param obs: 当前状态
        :param action: 采取的动作
        :param reward: 反馈的奖励
        :param next_obs: 下一状态
        :param terminated: 是否终止
        """
        self.memory.append(obs, action, reward, next_obs, terminated)

    def decay_epsilon(self):
        """
        衰减 epsilon（探索率），确保探索逐渐减少
        """
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def increase_beta(self):
        """
        增加经验回放的 beta 参数（仅适用于优先级经验回放）
        """
        self.memory.increase_beta()
