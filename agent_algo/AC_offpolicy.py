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
import torch.nn.functional as F
# 设备选择，默认使用 CPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Agent:
    """
    actor-critic 智能体类，用于训练或测试
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
        if args.asynchronous:
            self.obs_size = self.env.observation_space.shape[1]
            self.action_size = self.env.action_space[0].n  # 动作空间大小
        else:
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
        self.memory = MyDeque_AC(
            maxlength=args.MAX_EXPERIENCE,  # 经验回放区最大容量
            obs_size=self.obs_size,
            batch_size=args.batch_size
        )




        # self.use_prioritized = not args.stop_prio  # 是否使用优先经验回放（Prioritized Experience Replay）

        # 初始化 actor网络
        self.actor = Net(self.obs_size, self.action_size,final_ac=True).to(device)  # 训练用的主 Q 网络

        if self.mode == "test":
            state_dict = torch.load(weight_path)  # 加载测试模型权重
            self.actor.load_state_dict(state_dict)
        # 初始化critic
        self.critic = Net(self.obs_size, 1).to(device)  # 目标 Q 网络

        # 优化器 & 损失函数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=args.lr)

        # if self.use_prioritized:
        #     self.criterion = torch.nn.SmoothL1Loss(reduction="none")  # Huber 损失（用于优先级经验回放）
        # else:
        #     self.criterion = torch.nn.SmoothL1Loss()  # Huber 损失（标准 DQN）

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

        #根据返回的概率选择动作
        with torch.no_grad():
            probs=self.actor(obs)
            action=torch.multinomial(probs,1).item()
            self.probs=probs[action].item()
            return action


    def update(self):
        """
        使用经验回放进行 ac网络更新
        """
        # # 确保缓冲区中的经验足够一个 batch，否则跳过更新
        # if len(self.memory) < self.batch_size:
        #     self.training_error.append(0)
        #     return

        # 从经验回放缓冲区采样一个批次
        obs, action, reward, next_obs, terminated, w ,last_probs= self.memory.get_batch()


        # 计算当前 probs 值
        probs = self.actor(obs).gather(dim=1, index=action)  # 只选择执行的动作对应的 probs
        v_now=self.critic(obs)
        # 计算目标 Q 值
        with torch.no_grad():
            v_next=self.critic(next_obs)
            v_target = reward + self.gamma * v_next * (~terminated)  # 计算目标值
            delte = v_target - v_now
        loss_actor=torch.mean(-probs*delte.detach()/last_probs)
        loss_critic=torch.mean((v_now-v_target.detach())**2*probs.detach()/last_probs)    # wrong

        self.training_error.append(loss_actor.item())

        # 反向传播 + 梯度裁剪
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss_actor.backward()
        loss_critic.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()


    def record(self, obs, action, reward, next_obs, terminated):
        """
        将经验存储到经验回放缓冲区

        :param obs: 当前状态
        :param action: 采取的动作
        :param reward: 反馈的奖励
        :param next_obs: 下一状态
        :param terminated: 是否终止

        """
        self.memory.append(obs, action, reward, next_obs, terminated,self.probs)

    def decay_epsilon(self):
        """
        衰减 epsilon（探索率），确保探索逐渐减少
        """
        # self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def increase_beta(self):
        """
        增加经验回放的 beta 参数（仅适用于优先级经验回放）
        """
        # self.memory.increase_beta()
