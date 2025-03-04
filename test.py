import gymnasium as gym
from argument import args
import agent

# 创建环境实例，设置最大 episode 步数，并启用渲染模式（用于可视化）
env = gym.make(args.env_name, max_episode_steps=args.max_episode_steps, render_mode="human")

# 初始化智能体，设置为测试模式，并加载预训练的模型权重
agent = agent.Agent(env, mode="test", weight_path="logs/envCartPole-v1-20250304-122840/checkpoints/model_episode_400.pth")

# 初始化步数计数器
step = 0

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

    # 如果 episode 结束，打印总步数并退出循环
    if done:
        print(step)  # 打印当前 episode 的总步数
        break

# 关闭环境
env.close()