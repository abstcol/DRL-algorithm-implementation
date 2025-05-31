import sys
import os
# 添加父目录到路径，确保可以导入父目录中的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from argument import args
import importlib
import datetime
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")
from train_utils import *

def main():
    #创建惟一的时间戳，用于检查点目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir= f"experiments/env_{args.env_name}-algo_{args.algo_name}-{timestamp}"

    #创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)  # 自动创建目录\



    initialize_wandb(args,timestamp)

    env=create_env(args)


    #初始化智能体
    agent= Agent.Agent(env, mode="train")

    if args.asynchronous:
        train_agent_asynchronous(agent,env,args,checkpoint_dir,timestamp)
    else:train_agent(agent,env,args,checkpoint_dir,timestamp)


for _ in range(5):
    main()