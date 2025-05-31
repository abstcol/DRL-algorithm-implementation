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
            self.memory = MyDeque_Onpolicy(
                maxlength=args.MAX_EXPERIENCE,  # 经验回放区最大容量
                obs_size=self.obs_size,
                vec_num=args.vec_num,
                action_size=self.action_size
            )
        else:
            self.obs_size = self.env.observation_space.shape[0]  # 观察空间维度
            self.action_size = self.env.action_space.shape[0]  # 每次需要传递动作的数量
            # 经验回放缓冲区（Replay Buffer）
            self.memory = MyDeque_Onpolicy(
                maxlength=args.MAX_EXPERIENCE,  # 经验回放区最大容量
                obs_size=self.obs_size,
                vec_num=args.vec_num,
                action_size=self.action_size
            )

        # 训练超参数
        self.gamma = args.gamma  # 折扣因子
        self.lamb=args.lamb
        self.use_entropy=args.use_entropy
        self.entro_coe=args.entro_coe
        self.max_length=args.MAX_EXPERIENCE
        self.vec_num=args.vec_num
        self.batch=self.max_length*self.vec_num
        self.epsilon=1
        self.max_grad_norm=args.max_grad_norm
        # 初始化 actor网络
        self.actor = Gaussion_Net(self.obs_size, self.action_size,final_ac=False).to(device)  # 训练用的主 Q 网络

        if self.mode == "test":
            state_dict = torch.load(weight_path)  # 加载测试模型权重
            self.actor.load_state_dict(state_dict)
        # 初始化critic
        self.critic = Net(self.obs_size, 1).to(device)  # 目标 Q 网络

        # 优化器 & 损失函数
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=args.lr)


        # 训练相关
        self.training_error = []  # 记录损失值（用于日志记录）
        self.kl_divergence=[]     # 记录更新前后kl㪚度变化


    def get_action(self, obs):
        obs = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            mu, sigma = self.actor(obs)
            dist = torch.distributions.Normal(mu, sigma)  # 构造正态分布
            action = dist.sample()  # 采样连续动作

        return action.cpu().numpy()



    def get_gae(self,terminated,delte):

        gae=torch.zeros(self.batch).to(device)
        temp_gae=delte[-self.vec_num:]
        gae[-self.vec_num:]=temp_gae
        for i in range(1,self.max_length):
            temp_gae=self.lamb*self.gamma*temp_gae*(~terminated[-(i+1)*self.vec_num:-i*self.vec_num])+delte[-(i+1)*self.vec_num:-i*self.vec_num]
            gae[-(i+1)*self.vec_num:-i*self.vec_num]=temp_gae
        return gae




    def update(self):


        """
        使用经验回放进行 ac 网络更新
        """

        if self.memory.idx != self.memory.batch:
            return

        # 从经验回放缓冲区采样一个批次
        obs, action, reward, next_obs, terminated = self.memory.get_batch()


        # 计算当前 probs 值
        # 计算 Actor 损失
        mu, sigma = self.actor(obs)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(action).sum(dim=-1)

        v_now=self.critic(obs)
        # 计算目标 Q 值
        with torch.no_grad():
            v_next=self.critic(next_obs)
            v_target = reward + self.gamma * v_next * (~terminated)  # 计算目标值
            delte=(v_target-v_now).squeeze()
        gae=self.get_gae(  terminated.squeeze(),delte)
        loss_actor = -torch.mean(log_prob * gae)  # 计算策略梯度损失
        loss_critic=F.mse_loss(v_now,gae.unsqueeze(1)+v_now.detach())
        if self.use_entropy:
            # 计算熵增益
            entropy_loss = dist.entropy() # 计算策略的熵
            loss_actor-=self.entro_coe*entropy_loss.mean()

        self.training_error.append((loss_actor.item(),loss_critic.item()))

        # 反向传播 + 梯度裁剪
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        loss_actor.backward()
        loss_critic.backward()


        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),self.max_grad_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        #计算更新后的策略分布
        with torch.no_grad():
            mu_new, sigma_new = self.actor(obs)
            dist_new = torch.distributions.Normal(mu_new, sigma_new)

        # 计算 KL 散度
        kl_divergence = torch.distributions.kl_divergence(dist, dist_new).mean()
        self.kl_divergence.append(kl_divergence)


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
