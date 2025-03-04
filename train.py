import importlib
import gymnasium as gym
import torch
from tqdm import tqdm
from argument import args
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import agent
import os
def get_moving_avgs(arr,window,convolution_mode):
    """
    计算滑动平均值数组
    :param arr:输入数组
    :param window:滑动窗口大小
    :param convolution_mode:卷积模式
    :return:滑动平均后的数组
    """
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    )/window

#创建惟一的时间戳，用于日志和检查点目录
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"logs/env{args.env_name}-{timestamp}"  # 每次训练日志保存在不同目录
writer=SummaryWriter(log_dir)

#创建检查点目录
checkpoint_dir = os.path.join(log_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)  # 自动创建目录

#创建环境实例，并设置最大步数
env=gym.make(args.env_name,max_episode_steps=args.max_episode_steps)

#包装环境已记录环境统计信息
env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=args.episodes)

#初始化智能体
agent=agent.Agent(env,mode="train")
#步数计数器
step=1

#训练循环
for episode in tqdm(range(args.episodes)):

    #环境初始化，获取初始状态
    observation,info=env.reset()
    #记录当前episode起始状态
    last_step=step

    #episode内部循环
    while True:

        action=agent.get_action(observation)
        next_observation,reward,terminated,truncated,info=env.step(action)
        done=terminated or truncated
        if terminated:
            reward=-10
        agent.record(observation,action,reward,next_observation,terminated)
        observation=next_observation
        #每一步都进行一次更新
        agent.update()


        if done :
            #打印当前episode持续步数
            print(step-last_step)
            break
        step += 1
    #每个episode进行一次epsilon降
    agent.decay_epsilon()
    # episode结束后的处理
    if (episode + 1) % 50 == 0:  # 每隔50个episode保存
        checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode + 1}.pth")
        torch.save(agent.main_q.state_dict(), checkpoint_path)


env.close()

# 超参数记录（仅记录关键参数，避免表格太长）
hparams = {
    "env_name": args.env_name,
    "episodes": args.episodes,
    "max_steps": args.max_episode_steps,
    "batch_size": args.batch_size,
    "gamma": args.gamma,
    "alpha": args.alpha,
    "lr": args.lr,
    "init_eps": args.initial_epsilon,
    "final_eps": args.final_epsilon,
    "max_exp": args.MAX_EXPERIENCE,
}

# 记录超参数到 TensorBoard
writer.add_hparams(hparams, {})

# 记录详细参数文本
config_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
writer.add_text("Hyperparameter Config", config_text)

rolling_length=100
# print(env.length_queue)
# 计算平滑曲线
reward_moving_average = get_moving_avgs(env.return_queue, rolling_length, "valid")
length_moving_average = get_moving_avgs(env.length_queue, rolling_length, "valid")
training_error_moving_average = get_moving_avgs(agent.training_error, rolling_length, "valid")

# 记录到 TensorBoard
for i in range(len(reward_moving_average)):
    writer.add_scalar("Episode Reward", reward_moving_average[i], i)
    writer.add_scalar("Episode Length", length_moving_average[i], i)
    writer.add_scalar("Training Error", training_error_moving_average[i], i)

writer.close()


