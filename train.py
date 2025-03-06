import gymnasium as gym
import torch
from tqdm import tqdm
from argument import args
import importlib
import agent_algo
import datetime
import wandb
import os
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")

for _ in range(10):
    #创建惟一的时间戳，用于检查点目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_dir= f"experiements/env_{args.env_name}-algo_{args.algo_name}-{timestamp}"

    #创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)  # 自动创建目录\

    # 初始化 wandb
    wandb.init(
        project="RL-Training",
        config=vars(args),  # 记录所有超参数
        name=f"{args.env_name}{args.algo_name}-{timestamp}"
    )



    #创建环境实例，并设置最大步数
    env=gym.make(args.env_name,max_episode_steps=args.max_episode_steps)

    #包装环境已记录环境统计信息
    env=gym.wrappers.RecordEpisodeStatistics(env,buffer_length=args.episodes)

    #初始化智能体
    agent= Agent.Agent(env, mode="train")


    #步数计数器
    steps=1

    #训练循环
    for episode in tqdm(range(args.episodes)):

        #环境初始化，获取初始状态
        observation,info=env.reset()
        #记录当前episode起始状态
        last_steps=steps

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
                # print(step-last_step)
                break
            steps += 1
        step=steps-last_steps
        # 记录 `wandb` 指标
        wandb.log({
            "episode": episode + 1,
            "steps": step,
            "epsilon": agent.epsilon,  # 记录 epsilon 下降情况
            "loss":sum(agent.training_error[-step:])/step
        })

        #每个episode进行一次epsilon降
        agent.decay_epsilon()
        # episode结束后的处理
        if (episode + 1) % 50 == 0:  # 每隔50个episode保存
            checkpoint_path = os.path.join(checkpoint_dir, f"model_episode_{episode + 1}.pth")
            torch.save(agent.main_q.state_dict(), checkpoint_path)
            wandb.log_model(checkpoint_path,f"{args.env_name }_{args.algo_name}_{timestamp}",[f"average_length_{sum([env.length_queue[episode-i] for i in range(50)])/50}"])

    env.close()
    wandb.finish()