import gymnasium as gym
from argument import args

class WrapperMountainCar(gym.Wrapper):
    """
    自定义的 MountainCar 环境 Wrapper，用于修改奖励机制。
    """
    def __init__(self, env):
        """
        初始化 Wrapper 环境
        :param env: 原始的 MountainCar 环境
        """
        super().__init__(env)  # 调用父类构造函数

    def step(self, action):
        """
        执行一步环境交互，并修改奖励机制。
        :param action: 智能体选择的动作
        :return: 修改后的状态、奖励、终止标志、截断标志和额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # 执行原环境的 step 函数

        # 自定义奖励机制
        x, v = obs[0] + 0.5, obs[1]  # 位置加偏移量，速度

        # 如果速度大于阈值且采取加速动作，则奖励增加
        if v > 0.01 and action == 2:
            reward += v * 1000
        # 如果速度小于负阈值且采取减速动作，则奖励减少
        elif v < -0.01 and action == 0:
            reward -= v * 1000

        # 如果环境结束（智能体到达目标），奖励为 1000
        if terminated:
            reward = 1000

        return obs, reward, terminated, truncated, info  # 返回修改后的状态和奖励


class WrapperMountainCarContinueVec(gym.vector.VectorWrapper):
    """
    自定义的 Acrobot 环境 Wrapper，用于修改奖励机制。
    """
    def __init__(self, env):
        """
        初始化 Wrapper 环境
        :param env: 原始的 Acrobot 环境
        """
        super().__init__(env)  # 调用父类构造函数
        self.terminate_reward=args.terminate_reward

    def step(self, action):
        """
        执行一步环境交互，并修改奖励机制。
        :param action: 智能体选择的动作
        :return: 修改后的状态、奖励、终止标志、截断标志和额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # 执行原环境的 step 函数



        # 如果环境结束（智能体到达目标），奖励为 1000
        for i in range(args.vec_num):
            # # 自定义奖励机制
            # x, v = obs[i][0] + 0.5, obs[i][1]  # 位置加偏移量，速度
            #
            # # 如果速度大于阈值且采取加速动作，则奖励增加
            # if v > 0.01 and action[i] >0.01:
            #     reward[i] += (v+action[i]/20 ) * 100
            # # 如果速度小于负阈值且采取减速动作，则奖励减少
            # elif v < -0.01 and action[i] <-0.01:
            #     reward[i] -= (v+action[i]/20) * 100

            if terminated[i]:
                reward[i] = self.terminate_reward

        return obs, reward, terminated, truncated, info  # 返回修改后的状态和奖励






class WrapperAcrobot(gym.Wrapper):
    """
    自定义的 Acrobot 环境 Wrapper，用于修改奖励机制。
    """
    def __init__(self, env):
        """
        初始化 Wrapper 环境
        :param env: 原始的 Acrobot 环境
        """
        super().__init__(env)  # 调用父类构造函数
        self.terminate_reward=args.terminate_reward

    def step(self, action):
        """
        执行一步环境交互，并修改奖励机制。
        :param action: 智能体选择的动作
        :return: 修改后的状态、奖励、终止标志、截断标志和额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # 执行原环境的 step 函数

        # 自定义奖励机制
        cos1,sin1,cos2,sin2,v1,v2= obs[0] , obs[1],obs[2],obs[3],obs[4],obs[5]


        # 如果环境结束（智能体到达目标），奖励为 1000
        if terminated:
            reward = self.terminate_reward

        return obs, reward, terminated, truncated, info  # 返回修改后的状态和奖励

class WrapperAcrobotVec(gym.vector.VectorWrapper):
    """
    自定义的 Acrobot 环境 Wrapper，用于修改奖励机制。
    """
    def __init__(self, env):
        """
        初始化 Wrapper 环境
        :param env: 原始的 Acrobot 环境
        """
        super().__init__(env)  # 调用父类构造函数
        self.terminate_reward=args.terminate_reward

    def step(self, action):
        """
        执行一步环境交互，并修改奖励机制。
        :param action: 智能体选择的动作
        :return: 修改后的状态、奖励、终止标志、截断标志和额外信息
        """
        obs, reward, terminated, truncated, info = self.env.step(action)  # 执行原环境的 step 函数



        # 如果环境结束（智能体到达目标），奖励为 1000
        for i in range(args.vec_num):
            if terminated[i]:
                reward[i] = self.terminate_reward

        return obs, reward, terminated, truncated, info  # 返回修改后的状态和奖励



# 环境映射字典，便于在不同环境中应用自定义 Wrapper
class_map = {
    # "MountainCar-v0": [WrapperMountainCar,None],  # 对应的环境名称和 Wrapper 映射
    "Acrobot-v1":[WrapperAcrobot,WrapperAcrobotVec],
    "MountainCarContinuous-v0":[None,WrapperMountainCarContinueVec]
}
