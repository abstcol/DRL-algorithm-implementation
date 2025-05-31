import torch
from tqdm import tqdm
import datetime
from argument import args
import importlib
import wandb
import os
import imageio
from env import *
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")


def create_env(args):
    """创建 Gym 环境"""
    if args.produce_gif == False:
        # 创建环境实例，并设置最大步数
        if args.max_episode_steps == 0:
            env = gym.make(args.env_name, render_mode="human")
        else:
            env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode="human")
    else:
        if args.max_episode_steps == 0:
            env = gym.make(args.env_name, render_mode="rgb_array")
        else:
            env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=args.episodes)  # 记录环境统计信息
    return env

def load_agent(env):
    """加载预训练的智能体"""
    Agent = importlib.import_module(f"agent_algo.{args.algo_name}")
    run = wandb.init()
    # 获取artifact
    artifact = run.use_artifact(args.checkpoint_name, type='model')


    artifact_dir = os.path.join('artifacts', args.checkpoint_name)

    # 创建保存路径（如果不存在的话）
    os.makedirs(artifact_dir, exist_ok=True)

    # 下载artifact到指定的路径
    artifact.download(artifact_dir)

    model_files = os.listdir(artifact_dir)
    weight_path = os.path.join(artifact_dir, model_files[0])
    agent = Agent.Agent(env, mode="test", weight_path=weight_path)
    return agent


def save_gif(frames, args):
    """将收集到的帧保存为 GIF 动图，并生成动态文件名"""



    # 创建保存 GIF 的文件夹（如果不存在的话）
    save_dir = "gifs"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # 生成文件名，包含环境、算法、时间戳
    output_path = os.path.join(save_dir, f"{args.env_name}_{args.algo_name}_{timestamp}.gif")

    # 保存 GIF 动图
    with imageio.get_writer(output_path, mode='I', duration=0.05) as writer:
        for frame in frames:
            writer.append_data(frame)  # 将每一帧添加到 gif 动图中

    print(f"GIF saved to {output_path}")


def run_test_loop(agent, env):
    """运行一个完整的测试循环"""
    frames = []
    observation, info = env.reset()

    while True:
        # 根据当前状态选择动作（测试模式下使用贪婪策略）
        action = agent.get_action(observation)

        # 执行动作，获取下一个状态、奖励和终止标志
        next_observation, reward, terminated, truncated, info = env.step(action)

        # 判断 episode 是否结束
        done = terminated or truncated

        # 更新当前状态
        observation = next_observation

        # 渲染并将当前图像帧保存到列表中
        frame = env.render()  # 获取当前帧（rgb_array）
        frames.append(frame)  # 保存帧

        # 如果 episode 结束，退出循环
        if done:
            break

    return frames