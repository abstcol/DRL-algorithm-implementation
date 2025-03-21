from  argument import args
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import Dataset
import numpy as np
from collections import deque
from argument import args
import wandb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Net(nn.Module):
    """
    神经网络模型，用于近似q函数
    """
    def __init__(self,n_state,n_action,hidden_sizes=args.hidden_sizes,final_ac=None):
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
        if final_ac:
            layers.append(nn.Softmax(dim=-1))
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




class DuelingNet(nn.Module):
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
        super(DuelingNet,self).__init__()

        input_size=n_state
        layers=[]
        # 动态构建隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())  # 使用 LeakyReLU 作为激活函数
            input_size = hidden_size
        # 输出层
        layers.append(nn.Linear(input_size, n_action+1))

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
        ret=self.network(x)
        v=ret[:,0]
        a=ret[:,1:]
        q=a+v-torch.mean(a,dim=1)
        return  q





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
        self.min_p=10
        self.max_p=0.1
        self.beta=args.initial_beta
        self.prioepsilon=args.prioepsilon

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
        p=copy.deepcopy(self.memory_weights[index])
        w=(p/self.min_p)**(-self.beta)
        obs=self.memory_obs[index]
        action=self.memory_action[index]
        reward=self.memory_reward[index]
        next_obs=self.memory_next_obs[index]
        terminated=self.memory_terminated[index]
        return obs,action,reward,next_obs,terminated,w

    def get_index(self):
        """
        根据权重采样索引，该版本权重相等
        :return:索引序列，形状为(batch_size,1)
        """
        if self.full==True:
            return torch.multinomial(self.memory_weights,self.batch_size,replacement=True)
        return torch.multinomial(self.memory_weights[:self.idx],self.batch_size,replacement=True)

    def update(self, p):
        """
        修改每个样本的权重，依据是tderror，该版本未使用
        :param p:理想q与现实q差值的绝对值
        """
        for id,idx in enumerate(self.last_index):
            self.memory_weights[idx]= p[id] + 1e-6
            if p[id]<self.min_p:
                self.min_p=p[id]
            if p[id]>self.max_p:
                self.max_p=p[id]

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

        self.memory_weights[self.idx]=self.max_p   #初始权重

        self.idx+=1
        #循环队列
        if self.idx==self.maxlength:
            self.full=True
            self.idx=0
    def increase_beta(self):
        self.beta=min(self.beta*args.increase_beta,1)




class MyDeque_AC():
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
        self.memory_probs=torch.zeros((maxlength,1),dtype=torch.float32,device=device)
        self.memory_weights = torch.ones((maxlength,), dtype=torch.float32, device=device)
        self.idx=0  #当前写入位置
        self.full=False #存储区是否已满
        self.min_p=10
        self.max_p=0.1
        self.beta=args.initial_beta
        self.prioepsilon=args.prioepsilon

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
        p=copy.deepcopy(self.memory_weights[index])
        w=(p/self.min_p)**(-self.beta)
        obs=self.memory_obs[index]
        action=self.memory_action[index]
        reward=self.memory_reward[index]
        next_obs=self.memory_next_obs[index]
        terminated=self.memory_terminated[index]
        probs=self.memory_probs[index]
        return obs,action,reward,next_obs,terminated,w,probs

    def get_index(self):
        """
        根据权重采样索引，该版本权重相等
        :return:索引序列，形状为(batch_size,1)
        """
        if self.full==True:
            return torch.multinomial(self.memory_weights,self.batch_size,replacement=True)
        return torch.multinomial(self.memory_weights[:self.idx],self.batch_size,replacement=True)

    def update(self, p):
        """
        修改每个样本的权重，依据是tderror，该版本未使用
        :param p:理想q与现实q差值的绝对值
        """
        for id,idx in enumerate(self.last_index):
            self.memory_weights[idx]= p[id] + 1e-6
            if p[id]<self.min_p:
                self.min_p=p[id]
            if p[id]>self.max_p:
                self.max_p=p[id]

    def append(self,obs,action,reward,next_obs,terminated,probs):
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
        self.memory_probs[self.idx]=torch.tensor(probs,dtype=torch.float32,device=device)
        self.memory_weights[self.idx]=self.max_p   #初始权重

        self.idx+=1
        #循环队列
        if self.idx==self.maxlength:
            self.full=True
            self.idx=0
    def increase_beta(self):
        self.beta=min(self.beta*args.increase_beta,1)
