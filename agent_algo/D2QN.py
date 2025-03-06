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
# 自动检测是否有 GPU
#经过测试发现因为需要不断与环境交互，模型并不能很好利用gpu，反而会导致训练时长增加
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

class Net(nn.Module):
    """
    神经网络模型，用于近似q函数
    """
    def __init__(self,n_state,n_action,hidden_sizes=args.hidden_sizes):
        """
        初始化神经网络
        :param n_state:状态空间的size
        :param n_action:动作空间的size
        :param hidden_sizes:隐藏层的尺寸列表
        """
        super(Net,self).__init__()

        input_size=n_state
        layers=[]
        # 动态构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # 使用 LeakyReLU 作为激活函数
            input_size = hidden_size
        # 输出层
        layers.append(nn.Linear(input_size, n_action))

        self.network=nn.Sequential(*layers)
        # 直接应用初始化函数
        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        """
        对单个模块初始化，配合apply可达到递归初始化效果
        :param m:单个神经网络层
        """
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.zeros_(m.bias)

    def forward(self,x):
        """
        前向传播
        :param x:输入observation，形状为(batch_size,n_state)
        :return:输出为指定状态时各个动作的q alue 形状为(batch_size,n_action)
        """
        return self.network(x)

class MyDeque():
    def __init__(self,maxlength,obs_size,batch_size):
        """
        自定义的经验放回区，用于存储和采样经验，是一个循环队列结构，实质上直接使用dequeue也可实现，这里单独实现主要有两个目的：
        1.功能封装，代码更为简介
        2.想要直接将经验存储为tensor形式且保存在gpu中，减少cpu和gpu之间的通信次数，加速训练速度（事实证明没有用）
        :param maxlength:经验池的大小
        :param obs_size:状态空间的size
        :param batch_size:每一次采样的经验数量
        """

        self.maxlength=maxlength
        self.batch_size=batch_size

        #初始化存储经验的张量
        self.memory_obs = torch.zeros((maxlength, obs_size), dtype=torch.float32, device=device)
        self.memory_action = torch.zeros((maxlength, 1), dtype=torch.long, device=device)
        self.memory_reward = torch.zeros((maxlength, 1), dtype=torch.float32, device=device)
        self.memory_next_obs = torch.zeros((maxlength, obs_size), dtype=torch.float32, device=device)
        self.memory_terminated = torch.zeros((maxlength, 1), dtype=torch.bool, device=device)
        self.memory_weights = torch.ones((maxlength,), dtype=torch.float32, device=device)
        self.idx=0  #当前写入位置
        self.full=False #存储区是否已满

    def __len__(self):
        """
        返回当前缓冲区的经验数量，主要用于训练刚开始时经验数小于批量大小时拒绝采样
        :return:当前缓冲区的经验数量
        """
        if self.full ==True:
            return self.maxlength
        return self.idx

    def get_batch(self):
        """
        从缓冲区采样一个批次的数据
        :return:状态，动作，奖励，下一个状态，结束标志
        """
        index=self.get_index()
        self.last_index=index
        obs=self.memory_obs[index]
        action=self.memory_action[index]
        reward=self.memory_reward[index]
        next_obs=self.memory_next_obs[index]
        terminated=self.memory_terminated[index]
        return obs,action,reward,next_obs,terminated

    def get_index(self):
        """
        根据权重采样索引，该版本权重相等
        :return:索引序列，形状为(batch_size,1)
        """
        if self.full==True:
            return torch.multinomial(self.memory_weights,self.batch_size,replacement=True)
        return torch.multinomial(self.memory_weights[:self.idx],self.batch_size,replacement=True)

    def update(self,td_error):
        """
        修改每个样本的权重，依据是tderror，该版本未使用
        :param td_error:理想q与现实q差值的绝对值
        """
        for id,idx in enumerate(self.last_index):
            self.memory_weights[idx]=td_error[id]+1e-6

    def append(self,obs,action,reward,next_obs,terminated):
        """
        向缓冲区添加新的经验
        :param obs:当前状态
        :param action:采取的动作
        :param reward:获得的奖励
        :param next_obs:下一状态
        :param terminated:结束标志
        """
        self.memory_obs[self.idx]=torch.tensor(obs, dtype=torch.float32, device=device)
        self.memory_action[self.idx]=torch.tensor(action, dtype=torch.long, device=device)
        self.memory_reward[self.idx]=torch.tensor(reward, dtype=torch.float32, device=device)
        self.memory_next_obs[self.idx]=torch.tensor(next_obs, dtype=torch.float32, device=device)
        self.memory_terminated[self.idx] = torch.tensor(terminated, dtype=torch.bool, device=device)
        self.memory_weights[self.idx]=1e7   #初始权重
        self.idx+=1
        #循环队列
        if self.idx==self.maxlength:
            self.full=True
            self.idx=0

class Agent:
    """
    DQN智能体类，用于训练或测试
    """
    def __init__(
            self,
            env:gym.Env,
            mode:str,
            weight_path="none",
    ):
        """
        初始化智能体
        :param env:环境实例，通过该实例可以方便获得例如状态尺寸等信息
        :param mode:模式("train" or "test")
        :param weight_path:模型权重路径（仅在训练模式下使用）
        """
        self.env=env
        self.mode=mode
        self.obs_size=self.env.observation_space.shape[0]
        self.action_size=  self.env.action_space.n
        self.gamma=args.gamma
        self.alpha=args.alpha
        self.epsilon=args.initial_epsilon
        self.final_epsilon=args.final_epsilon
        self.epsilon_decay=args.decay_epsilon
        self.tau=args.tau
        self.batch_size = args.batch_size
        #初始化经验回放区
        self.memory=MyDeque(maxlength=args.MAX_EXPERIENCE,obs_size=self.obs_size,batch_size=args.batch_size)

        #初始化q网络和目标网络
        self.main_q=Net(self.obs_size,self.action_size).to(device)
        if self.mode=="test":
            state_dict=torch.load(weight_path)
            self.main_q.load_state_dict(state_dict)
        self.target_q=Net(self.obs_size,self.action_size).to(device)
        self.target_q.load_state_dict(self.main_q.state_dict())

        #优化器和损失函数
        self.optimizer=torch.optim.Adam(self.main_q.parameters(),lr=args.lr)
        self.criterion=torch.nn.SmoothL1Loss()
        #更新计数器
        self.id=0
        #存储训练误差，用于日志存储
        self.training_error=[]

    def get_action(self, obs)->int:
        """
        根据当前状态选择动作，训练模式所用策略为epsilon greedy，
        测试模式所用策略为greedy
        :param obs:当前状态
        :return:下一个动作
        """
        obs=torch.from_numpy(obs).to(device)

        if self.mode=="train":
            with torch.no_grad():
                #有spsilon的概率探索
                if np.random.random()<self.epsilon:
                    return self.env.action_space.sample()
            #利用
            action=torch.argmax(self.main_q(obs)).item()
            return action
        #测试模式下为greedy 策略
        if self.mode=="test":
            action = torch.argmax(self.main_q(obs)).item()
            return action

    def update(self):
        """
        更新q网络
        """
        #当缓冲区中经验数不足一个批量时，不进行更新
        if len(self.memory)<self.batch_size:
            self.training_error.append(0)
            return

        #从缓冲区中获得一个批次的数据
        obs,action,reward,next_obs,terminated=self.memory.get_batch()
        #计算当前q值
        q=self.main_q(obs)
        q=q.gather(dim=1,index=action)

        #计算目标q值
        with torch.no_grad():

            index_q_t = self.main_q(next_obs).argmax(dim=1, keepdims=True)
            q_t=self.target_q(next_obs).gather(1,index_q_t)
            q_t=self.gamma*torch.max(q_t,dim=1,keepdim=True).values*(~terminated)+reward
            q_pred = q.squeeze(1)
            td_error = torch.abs(q_t.squeeze(1) - q_pred)  # 计算时序差分误差，该版本为实现 shape: (batch_size,)

        #计算损失
        loss=self.criterion(q,q_t).float()
        self.training_error.append(loss.item())

        #进行优化
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.main_q.parameters():
            param.grad.data.clamp_(-1, 1)  # 限制梯度值在 -1 到 1 的范围内，这是防止梯度值变得过大或过小、导致训练不稳定
        self.optimizer.step()
        #软更新目标网络
        self.id += 1
        if self.id%4==0:
            for target_param, local_param in zip(self.target_q.parameters(), self.main_q.parameters()):  # <- soft update
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def record(self,obs,action,reward,next_obs,terminated):
        """
        记录经验到缓冲区
        :param obs:当前状态
        :param action:采取的动作
        :param reward:获得的奖励
        :param next_obs:下一状态
        :param terminated:结束标志
        """
        self.memory.append(obs,action,reward,next_obs,terminated)

    def decay_epsilon(self):
        """
        衰减探索率
        """
        self.epsilon=max(self.final_epsilon,self.epsilon*self.epsilon_decay)
