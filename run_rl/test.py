import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from argument import args


import importlib
import imageio
import wandb
# 添加父目录到路径，确保可以导入父目录中的模块

from test_utils import *
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")

# 创建环境和加载智能体
env = create_env(args)
agent = load_agent(env)

# 运行测试循环并获取帧数据
frames = run_test_loop(agent, env)

# 关闭环境
env.close()

# 如果需要生成 gif，保存动图
if args.produce_gif:
    save_gif(frames,args)