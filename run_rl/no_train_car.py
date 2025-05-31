from tqdm import tqdm
from argument import args
import importlib
import datetime
import wandb
import os
from env import *
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")
from train_utils import *

for _ in range(5):
    #创建惟一的时间戳，用于检查点目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir= f"experiements/env_{args.env_name}-algo_{args.algo_name}-{timestamp}"
    #创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)  # 自动创建目录\
    initialize_wandb(args)
    env=create_env(args)

    for episode in tqdm(range(args.episodes)):

        # 初始化环境
        observation, info = env.reset()
        action_delay = 0  # 控制动作切换的冷却时间

        while True:
            # 选择动作
            if action_delay == 0:
                velocity = observation[1]  # 速度
                position = observation[0]  # 位置

                if velocity > 0.01:
                    action = 2  # 向右推
                elif velocity < -0.01:
                    action = 0  # 向左推
                elif position > -0.5:
                    action = 0
                    action_delay = 5  # 设置冷却时间
                else:  # position <= -0.5
                    action = 2
                    action_delay = 5

            else:
                action_delay -= 1  # 冷却时间递减

            next_observation,reward,terminated,truncated,info=env.step(action)
            done=terminated or truncated
            observation=next_observation
            #每一步都进行一次更新
            if done :
                #打印当前episode持续步数
                # print(step-last_step)
                break
        step=env.length_queue[-1]
        # 记录 `wandb` 指标
        log_metrics(episode,env,step=step)

    env.close()
    wandb.finish()