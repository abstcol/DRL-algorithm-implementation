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
    ac 智能体类，用于训练或测试
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
            self.obs_size=self.env.observation_space.shape[1]
            self.action_size = self.env.action_space.shape[1]# 每次需要传递动作的数量
            # 经验回放缓冲区（Replay Buffer）
            self.memory = MyDeque_Continuous(
                maxlength=args.vec_num,  # 经验回放区最大容量
                obs_size=self.obs_size,
                batch_size=args.vec_num,
                action_size=self.action_size
            )
        else:
            self.obs_size = self.env.observation_space.shape[0]  # 观察空间维度
            self.action_size = self.env.action_space.shape[0]  # 每次需要传递动作的数量
            # 经验回放缓冲区（Replay Buffer）
            self.memory = MyDeque_Continuous(
                maxlength=1,  # 经验回放区最大容量
                obs_size=self.obs_size,
                batch_size=1,
                action_size=self.action_size
            )

        #  训练超参数
        self.gamma = args.gamma  # 折扣因子
        self.alpha = args.alpha  # 经验回放的优先级调整参数
        self.epsilon = args.initial_epsilon  # 初始探索率
        self.final_epsilon = args.final_epsilon  # 最小探索率
        self.epsilon_decay = args.decay_epsilon  # 探索率衰减系数
        self.tau = args.tau  # 目标网络软更新参数
        self.batch_size = args.batch_size  # 训练批量大小
        self.ppo_epsilon=args.ppo_epsilon
        self.update_time=args.update_time

        # # 经验回放缓冲区（Replay Buffer）
        # self.memory = MyDeque(
        #     maxlength=args.MAX_EXPERIENCE,  # 经验回放区最大容量
        #     obs_size=self.obs_size,
        #     batch_size=args.batch_size
        # )




        # self.use_prioritized = not args.stop_prio  # 是否使用优先经验回放（Prioritized Experience Replay）

        # 初始化 actor网络
        self.actor = Gaussion_Net(self.obs_size, self.action_size,final_ac=False).to(device)  # 训练用的主 Q 网络
        self.actor_old=Gaussion_Net(self.obs_size, self.action_size,final_ac=False).to(device)  # 用于计算ratio，衡量分布间差距
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
        self.kl_divergence=[]     # 记录更新前后kl㪚度变化
        self.use_entropy=args.use_entropy
        self.entro_coe=args.entro_coe

    def get_action(self, obs):
        obs = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            mu, sigma = self.actor(obs)
            dist = torch.distributions.Normal(mu, sigma)  # 构造正态分布
            action = dist.sample()  # 采样连续动作
            action = torch.clamp(action, -1, 1)  # 限制动作范围
        return action.cpu().numpy()

    def actor_learn(self,obs,action,log_prob_old,delte,i):
        mu, sigma = self.actor(obs)
        dist = torch.distributions.Normal(mu, sigma)
        # 计算当前 probs 值

        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算 log π(a|s)
        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * delte.detach()
        surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * delte.detach()
        # 计算 Actor 损失
        loss_actor = -torch.mean(torch.min(surr1, surr2))

        if self.use_entropy:
            # 计算熵增益
            entropy_loss = dist.entropy()  # 计算策略的熵
            loss_actor -= self.entro_coe * entropy_loss.mean()
        self.training_error.append(loss_actor.item())
        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        self.actor_optimizer.step()
        if i==args.update_time-1:
            # 计算更新后的策略分布
            with torch.no_grad():
                mu_new, sigma_new = self.actor(obs)
                dist_new = torch.distributions.Normal(mu_new, sigma_new)

            # 计算 KL 散度
            kl_divergence = torch.distributions.kl_divergence(dist, dist_new).mean()
            self.kl_divergence.append(kl_divergence)

    def critic_learn(self,obs,next_obs,reward,terminated,i):
        v_now = self.critic(obs)
        # 计算目标 Q 值
        with torch.no_grad():
            v_next = self.critic(next_obs)
            v_target = reward + self.gamma * v_next * (~terminated)  # 计算目标值

        loss_critic = F.mse_loss(v_now, v_target.detach())

        # 反向传播 + 梯度裁剪

        self.critic_optimizer.zero_grad()

        loss_critic.backward()

        self.critic_optimizer.step()

    def update(self):
        """
        使用经验回放进行 ac 网络更新
        """
        # # 确保缓冲区中的经验足够一个 batch，否则跳过更新
        # if len(self.memory) < self.batch_size:
        #     self.training_error.append(0)
        #     return

        # 从经验回放缓冲区采样一个批次

        self.actor_old.load_state_dict(self.actor.state_dict())


        obs, action, reward, next_obs, terminated, w = self.memory.get_batch()

        v_now = self.critic(obs)
        # 计算目标 v 值
        with torch.no_grad():

            v_next = self.critic(next_obs)
            v_target = reward + self.gamma * v_next * (~terminated)  # 计算目标值
            delte = (v_target - v_now).squeeze()
            mu_old,sigma_old=self.actor_old(obs)
            dist_old=torch.distributions.Normal(mu_old,sigma_old)
            log_prob_old=dist_old.log_prob(action).sum(-1).detach()      # N*action_size

        # 计算当前 probs 值
        # 计算 Actor 损失
        mu, sigma = self.actor(obs)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算 log π(a|s)


        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * delte.detach()
        surr2 = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon) * delte.detach()
        # loss_actor = -torch.mean(torch.min(surr1,surr2))  # 计算策略梯度损失
        loss_actor = -torch.mean(log_prob*delte)  # 计算策略梯度损失

        loss_critic = F.mse_loss(v_now, v_target.detach())
        if self.use_entropy:
            # 计算熵增益
            entropy_loss = dist.entropy()  # 计算策略的熵
            loss_actor -= self.entro_coe * entropy_loss.mean()

        self.training_error.append((loss_actor.item(),loss_critic.item()))

        # 反向传播 + 梯度裁剪
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss_actor.backward()
        loss_critic.backward()

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        # 计算更新后的策略分布
        with torch.no_grad():
            mu_new, sigma_new = self.actor(obs)
            dist_new = torch.distributions.Normal(mu_new, sigma_new)

        # 计算 KL 散度
        kl_divergence = torch.distributions.kl_divergence(dist, dist_new).mean()
        self.kl_divergence.append(kl_divergence)

        # for i in range(self.update_time):
        #     self.actor_learn( obs, action, log_prob_old, delte,i)
        #
        # for i in range(self.update_time):
        #     self.critic_learn(obs,next_obs,reward,terminated,i)





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
        # self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def increase_beta(self):
        """
        增加经验回放的 beta 参数（仅适用于优先级经验回放）
        """
        # self.memory.increase_beta()
