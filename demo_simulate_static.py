# -*- coding: utf-8 -*-
"""
静态路径规划示例
 Created on Wed Mar 13 2024 18:18:07
 Modified on 2024-3-13 18:18:07
 
 @auther: HJ https://github.com/zhaohaojie1998
"""
#
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.close('all')


'''环境实例化'''
from path_plan_env import StaticPathPlanning, NormalizedActionsWrapper
env = NormalizedActionsWrapper(StaticPathPlanning())


'''策略加载'''
import onnxruntime as ort
policy = ort.InferenceSession("./path_plan_env/policy_static.onnx")


'''仿真LOOP'''
from copy import deepcopy

MAX_EPISODE = 20        # 总的训练/评估次数
for episode in range(MAX_EPISODE):
    ## 获取初始观测
    obs = env.reset()
    ## 进行一回合仿真
    for steps in range(env.max_episode_steps):
        # 可视化
        env.render()
        # 决策
        obs = obs.reshape(1, *obs.shape)                      # (*shape, ) -> (1, *shape, )
        act = policy.run(['action'], {'observation': obs})[0] # return [action, ...]
        act = act.flatten()                                   # (1, dim, ) -> (dim, )
        # 仿真
        next_obs, _, _, info = env.step(act)
        # 回合结束
        if info["terminal"]:
            print('回合: ', episode,'| 状态: ', info,'| 步数: ', steps) 
            break
        else:
            obs = deepcopy(next_obs)
    #end for
#end for






r'''
#             ⠰⢷⢿⠄
#         ⠀⠀⠀⠀⠀⣼⣷⣄
#         ⠀⠀⣤⣿⣇⣿⣿⣧⣿⡄
#         ⢴⠾⠋⠀⠀⠻⣿⣷⣿⣿⡀
#         🏀   ⢀⣿⣿⡿⢿⠈⣿
#          ⠀⠀⢠⣿⡿⠁⢠⣿⡊⠀⠙
#          ⠀⠀⢿⣿⠀⠀⠹⣿
#           ⠀⠀⠹⣷⡀⠀⣿⡄
#            ⠀⣀⣼⣿⠀⢈⣧ 
#
#       你。。。干。。。嘛。。。
#       哈哈。。唉哟。。。
'''