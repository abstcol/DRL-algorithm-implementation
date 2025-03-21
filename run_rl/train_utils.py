import gymnasium.wrappers
import torch
from tqdm import tqdm

import importlib
import wandb
import os
import numpy as np
import datetime
from env import *
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")

def create_env(args):
    """创建 Gym 环境，并应用包装"""
    Wrapper = class_map.get(args.env_name, None)
    if not args.asynchronous:
        env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps) if args.max_episode_steps else gym.make(args.env_name)
        if Wrapper:
            env = Wrapper[0](env)  # 应用环境包装
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=args.episodes)  # 记录环境统计信息
    else:
        if args.max_episode_steps:
            env=gym.vector.AsyncVectorEnv([lambda:gym.make(args.env_name, max_episode_steps=args.max_episode_steps) for
                                           _ in range(args.vec_num)],autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        else:
            env = gym.vector.AsyncVectorEnv([lambda: gym.make(args.env_name) for _ in range(args.vec_num)],autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
        if Wrapper:
            env=Wrapper[1](env)
        env=gymnasium.wrappers.vector.RecordEpisodeStatistics(env)
    return env
def initialize_wandb(args, timestamp):
    """初始化 wandb 以进行日志记录"""


    wandb.init(
        project="RL-Training_a3c",
        config=vars(args),
        name=f"{args.env_name}{args.algo_name}-{timestamp}"
    )
def save_checkpoint(agent, checkpoint_dir, episode):
    """保存模型检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode + 1}.pth")
    torch.save(agent.actor.state_dict(), checkpoint_path)
    return checkpoint_path


def log_metrics(episode, env, agent=None, step=None):
    """记录 wandb 训练过程中的指标"""
    if agent:
        metrics = {
            "episode": episode + 1,
            "reward": env.return_queue[-1],
            "step": step,
            "epsilon": agent.epsilon,
            "loss": sum(agent.training_error[-step:]) / step if step > 0 else 0,
            "beta": agent.memory.beta
        }

        wandb.log(metrics)
    else:
        metrics={
            "episode": episode + 1,
            "reward": env.return_queue[-1],
            "step": step,
        }
        wandb.log(metrics)

def get_train_score(env,args):
    average_step=np.mean(env.length_queue).item()
    shortest_step=min(env.length_queue)
    return average_step,shortest_step

def train_agent(agent, env, args, checkpoint_dir, timestamp):
    """训练智能体"""
    for episode in tqdm(range(args.episodes)):
        observation, info = env.reset()
        while True:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.record(observation, action, reward, next_observation, terminated)
            observation = next_observation
            agent.update()

            if done:
                break

        step = env.length_queue[-1]
        log_metrics(episode, env, agent, step)
        agent.decay_epsilon()
        agent.increase_beta()

        if (episode + 1) % 50 == 0:
            checkpoint_path = save_checkpoint(agent, checkpoint_dir, episode)
            avg_reward = sum([env.return_queue[episode - i] for i in range(50)]) / 50
            wandb.log_model(checkpoint_path, f"{args.env_name}_{args.algo_name}_{timestamp}", [f"average_reward_{avg_reward}"])



    env.close()
    wandb.finish()


def train_agent_asynchronous(agent, env, args, checkpoint_dir, timestamp):
    """训练智能体"""
    observation, info = env.reset()
    episode_start = env.prev_dones
    for episode in tqdm(range(args.episodes)):


        while True:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)


            for i in range(args.vec_num):
                if  episode_start[i]:
                    continue
                agent.record(observation[i], action[i], reward[i], next_observation[i], terminated[i])

            observation = next_observation

            agent.update()
            episode_start=env.prev_dones
            if env.prev_dones[0]:
                break

        step = env.length_queue[-1]
        log_metrics(episode, env, agent, step)
        agent.decay_epsilon()
        agent.increase_beta()

        if (episode + 1) % 50 == 0:
            checkpoint_path = save_checkpoint(agent, checkpoint_dir, episode)
            avg_reward = sum([env.return_queue[ - i] for i in range(50)]) / 50
            wandb.log_model(checkpoint_path, f"{args.env_name}_{args.algo_name}_{timestamp}", [f"average_reward_{avg_reward}"])



    env.close()
    wandb.finish()




def sweep_train_agent(agent, env, args, checkpoint_dir, timestamp):
    """训练智能体"""
    for episode in tqdm(range(args.episodes)):
        observation, info = env.reset()
        while True:
            action = agent.get_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.record(observation, action, reward, next_observation, terminated)
            observation = next_observation
            agent.update()

            if done:
                break

        step = env.length_queue[-1]
        log_metrics(episode, env, agent, step)
        agent.decay_epsilon()
        agent.increase_beta()

        if (episode + 1) % 50 == 0:
            checkpoint_path = save_checkpoint(agent, checkpoint_dir, episode)
            avg_reward = sum([env.return_queue[episode - i] for i in range(50)]) / 50
            wandb.log_model(checkpoint_path, f"{args.env_name}_{args.algo_name}_{timestamp}", [f"average_reward_{avg_reward}"])


    average_step,shortest_step=get_train_score(env,args)
    env.close()
    return average_step
