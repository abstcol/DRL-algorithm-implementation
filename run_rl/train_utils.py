import torch
from tqdm import tqdm
from argument import args
import importlib
import wandb
import os
from env import *
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")

def create_env(args):
    """创建 Gym 环境，并应用包装"""
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps) if args.max_episode_steps else gym.make(args.env_name)
    Wrapper = class_map.get(args.env_name, None)
    if Wrapper:
        env = Wrapper(env)  # 应用环境包装
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=args.episodes)  # 记录环境统计信息
    return env
def initialize_wandb(args, timestamp):
    """初始化 wandb 以进行日志记录"""


    wandb.init(
        project="RL-Training",
        config=vars(args),
        name=f"{args.env_name}{args.algo_name}-{timestamp}"
    )
def save_checkpoint(agent, checkpoint_dir, episode):
    """保存模型检查点"""
    checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode + 1}.pth")
    torch.save(agent.main_q.state_dict(), checkpoint_path)
    return checkpoint_path


def log_metrics(episode, env, agent=None, step=None):
    """记录 wandb 训练过程中的指标"""
    if agent:
        metrics = {
            "episode": episode + 1,
            "reward": env.return_queue[episode],
            "step": step,
            "epsilon": agent.epsilon,
            "loss": sum(agent.training_error[-step:]) / step if step > 0 else 0,
            "beta": agent.memory.beta
        }

        wandb.log(metrics)
    else:
        metrics={
            "episode": episode + 1,
            "reward": env.return_queue[episode],
            "step": step,
        }
        wandb.log(metrics)

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
