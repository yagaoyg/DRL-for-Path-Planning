import matplotlib.pyplot as plt
from shapely import geometry as geo
from matplotlib.patches import Polygon as plt_polygon
import numpy as np
from shapely.plotting import plot_polygon

class MAP:
    size = [[-13.0, -13.0], [13.0, 13.0]] # x, z最小值; x, z最大值
    start_pos = [-10, -5]                   # 起点坐标
    end_pos = [2.5, 9]                    # 终点坐标
    obstacles = [                         # 障碍物, 要求为 geo.Polygon 或 带buffer的 geo.Point/geo.LineString
        geo.Point(-3, 3.5).buffer(3.5),
        geo.Point(5, 2.5).buffer(3),
        geo.Point(-6, -5).buffer(3),
        geo.Point(6, -5).buffer(3),
        
        geo.Polygon([(-10, 0), (-10, 5), (-7.5, 5), (-7.5, 0)]),
        geo.Polygon([(2, 8), (2, 10), (10, 10), (10, 8)]),
        geo.Polygon([(-1, -10), (-1, -6), (1, -4), (1, -10)]),
        
        geo.Polygon([(-14, -14), (-13, -14), (-13, 14), (-14, 14)]),
        geo.Polygon([(14, -14), (13, -14), (13, 14), (14, 14)]),
        geo.Polygon([(-14, 13), (-14, 14), (14, 14), (14, 13)]),
        geo.Polygon([(-14, -13), (-14, -14), (14, -14), (14, -13)]),
    ]
    
    moving_rect = geo.Polygon([(-2, -2), (-1, -2), (-1, -1), (-2, -1)])  # 初始矩形

    @classmethod
    def show(cls):
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.close('all')
        fig, ax = plt.subplots()
        ax.clear()
        cls.plot(ax)

        # 绘制矩形
        cls.plot_moving_rectangle(ax)

        ax.legend(loc='best').set_draggable(True)
        plt.show(block=True)

    @classmethod
    def plot(cls, ax, title='Map'):
        ax.clear()
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.grid(alpha=0.3, ls=':')
        ax.set_xlim(cls.size[0][0], cls.size[1][0])
        ax.set_ylim(cls.size[0][1], cls.size[1][1])
        ax.invert_yaxis()

        # 绘制障碍物
        for o in cls.obstacles:
            plot_polygon(o, ax=ax, facecolor='w', edgecolor='k', add_points=False)
            
    @classmethod
    def plot_moving_rectangle(cls, ax):
        # 绘制矩形，更新位置
        # 假设我们每次移动矩形1个单位
        dx, dy = 0.1, 0.1  # 移动步长

        # 更新矩形的坐标
        new_coords = [(p[0] + dx, p[1] + dy) for p in cls.moving_rect.exterior.coords]
        updated_rect = geo.Polygon(new_coords)
        cls.moving_rect = updated_rect

        # 绘制更新后的矩形
        x, y = updated_rect.exterior.xy
        ax.fill(x, y, alpha=0.5, color='red', label="移动矩形")
        
    @staticmethod
    def plot_polygon(polygon, ax, **kwargs):
        """绘制多边形"""
        x, y = polygon.exterior.xy
        ax.fill(x, y, **kwargs)
            
# 执行显示地图和移动矩形
MAP.show()
