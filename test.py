import gymnasium as gym
from argument import args
import os
import importlib
import imageio
Agent=importlib.import_module(f"agent_algo.{args.algo_name}")

if args.produce_gif==False:
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode="human")
# 创建环境实例，设置最大 episode 步数，并启用渲染模式（用于可视化）
else:
    env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode="rgb_array")


import wandb
run = wandb.init()
artifact = run.use_artifact(args.checkpoint_name, type='model')
artifact_dir = artifact.download()
model_files = os.listdir(artifact_dir)  # 列出目录下所有文件
weight_path=os.path.join(artifact_dir,model_files[0])
# 初始化智能体，设置为测试模式，并加载预训练的模型权重
agent = Agent.Agent(env, mode="test", weight_path=weight_path)

# 初始化步数计数器
step = 0

# 初始化存储帧的列表
frames = []

# 重置环境，获取初始状态
observation, info = env.reset()

# 测试循环
while True:
    # 根据当前状态选择动作（测试模式下使用贪婪策略）
    action = agent.get_action(observation)

    # 执行动作，获取下一个状态、奖励和终止标志
    next_observation, reward, terminated, truncated, info = env.step(action)

    # 判断 episode 是否结束
    done = terminated or truncated

    # 更新当前状态
    observation = next_observation

    # 更新总步数
    step += 1

    # 渲染并将当前图像帧保存到列表中
    frame = env.render()  # 获取当前帧（rgb_array）
    frames.append(frame)  # 保存帧

    # 如果 episode 结束，打印总步数并退出循环
    if done:
        print(f"Episode finished in {step} steps.")  # 打印当前 episode 的总步数
        break

# 关闭环境
env.close()

if args.produce_gif==True:
    # 保存动图（gif）
    output_path = "test_episode.gif"
    with imageio.get_writer(output_path, mode='I', duration=0.05) as writer:
        for frame in frames:
            writer.append_data(frame)  # 将每一帧添加到 gif 动图中

    print(f"GIF saved to {output_path}")
