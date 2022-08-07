import numpy as np
import random
import math
import time
from tkinter import *


class GUI(object):
    def __init__(self, main_window, width=800, height=800):
        main_window.title("PSO算法可视化")
        main_window.geometry(f'{width}x{height}')
        self.canvas = Canvas(main_window, width=width, height=height)
        self.canvas.pack()

    def creat_Prey(self, centre_x, centre_y, gap=15):  # 制造猎物
        prey = self.canvas.create_oval(centre_x - gap, centre_y - gap, centre_x + gap, centre_y + gap,
                                       fill='black')
        return prey

    def creat_Predator(self, centre_x, centre_y, gap=4):  # 制造猎人
        predator = self.canvas.create_oval(centre_x - gap, centre_y - gap, centre_x + gap, centre_y + gap, fill='blue',
                                           outline='yellow')
        return predator


class PSO_model:
    def __init__(self, c1, c2, N, D, M):
        self.ws = 0.8
        self.we = 0.6
        self.c1 = c1
        self.c2 = c2
        self.r_max = 0.9
        self.r_min = 0.5
        self.N = N  # 初始化种群数量个数
        self.D = D  # 搜索空间维度
        self.M = M  # 迭代的最大次数
        self.x = np.zeros((self.N, self.D))  # 粒子的初始位置
        self.v = np.zeros((self.N, self.D))  # 粒子的初始速度
        self.pbest = np.zeros((self.N, self.D))  # 个体最优值初始化
        self.gbest = np.zeros((1, self.D))  # 种群最优值
        self.p_fit = np.zeros(self.N)  # 个体最优值
        self.fit = 1e8  # 初始化全局最优适应度
        self.sign = 0

        # 可视化
        self.main_window = Tk()
        self.PSO = GUI(self.main_window)

        # 初始化 猎物 猎手
        self.x_gui = []
        self.y = []
        self.y.append(random.uniform(400, 450))
        self.y.append(random.uniform(400, 450))
        self.y_gui = self.PSO.creat_Prey(self.y[0], self.y[1])

    # 适应度函数（求最小化问题）
    def function(self, x):
        x1 = x[0]
        x2 = x[1]
        y1 = self.y[0]
        y2 = self.y[1]
        return math.sqrt((x1 - y1) ** 2 + (x2 - y2) ** 2)

    # 初始化种群
    def init_pop(self):
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(20, 100)
                self.v[i][j] = random.uniform(0, 3)
            self.x_gui.append(self.PSO.creat_Predator(self.x[i][0], self.x[i][1]))  # 可视化
            self.pbest[i] = self.x[i]  # 初始化个体的最优值
            aim = self.function(self.x[i])  # 计算个体的适应度值
            self.p_fit[i] = aim  # 初始化个体的最优位置
            if aim < self.fit:  # 对个体适应度进行比较，计算出最优的种群适应度
                self.fit = aim
                self.gbest = self.x[i]

    # 更新粒子的位置与速度
    def update(self):
        t = 0
        while t != self.M:  # 在迭代次数M内进行循环
            self.w = self.ws + (self.we - self.ws) * (15 / self.function(self.gbest))  # 根据gbest更新权重
            self.r1 = self.r_min + (self.r_max - self.r_min) * (t / self.M)
            self.r2 = self.r_min + (self.r_max - self.r_min) * (t / self.M)
            for i in range(self.N):  # 对所有种群进行一次循环
                aim = self.function(self.x[i])  # 计算一次目标函数的适应度
                if aim < self.p_fit[i]:  # 比较适应度大小，将小的值给个体最优
                    self.p_fit[i] = aim
                    self.pbest[i] = self.x[i]
                    if self.p_fit[i] < self.fit:  # 如果是个体最优再将和全体最优进行对比
                        self.gbest = self.x[i]
                        self.fit = self.p_fit[i]

            for i in range(self.N):  # 更新粒子的速度和位置
                self.v[i] = self.w * self.v[i] + self.c1 * self.r1 * (self.pbest[i] - self.x[i]) + self.c2 * self.r2 * (
                        self.gbest - self.x[i])
                self.PSO.canvas.move(self.x_gui[i], self.v[i][0], self.v[i][1])
                self.main_window.update()
                time.sleep(0.03)
                self.x[i] = self.x[i] + self.v[i]
                # # self.x[i] = self.keep_distance(self.x[i], i)
                aim = self.function(self.x[i])
                if aim < 15:
                    escape = self.y - self.x[i]
                    escape = 5 * self.normalize(escape)
                    self.y = self.y + escape
                    self.PSO.canvas.move(self.y_gui, escape[0], escape[1])
                    self.main_window.update()
                    time.sleep(0.02)
                    self.fit = 1e8  # 初始化全局最优适应度
                    for i in range(self.N):  # 对所有种群进行一次循环
                        aim = self.function(self.x[i])  # 计算一次目标函数的适应度
                        self.p_fit[i] = aim
                        self.pbest[i] = self.x[i]
                        if self.p_fit[i] < self.fit:  # 如果是个体最优再将和全体最优进行对比
                            self.gbest = self.x[i]
                            self.fit = self.p_fit[i]
            for i in range(self.N):
                aim = self.function(self.x[i])  # 计算一次目标函数的适应度
                if aim >= 25:
                    break
                if i == self.N - 1:
                    self.main_window.title("PSO算法可视化   以达到目标状态 迭代次数: %d" % t)
                    self.sign = 1
            if self.sign:
                break
            self.main_window.title("PSO算法可视化 迭代次数: %d" % t)
            # 如果所有点都在15-20范围内 则追寻成功
            t = t + 1

    def normalize(self, escape):  # 猎物跑路的速度标准化（避免跑太快）
        if escape[0] > 0:
            escape[0] = 1
        else:
            escape[0] = -1
        if escape[1] > 0:
            escape[1] = 1
        else:
            escape[1] = -1
        return escape

    # 实时更新
    def loop(self):
        self.main_window.mainloop()


if __name__ == '__main__':
    # w,c1,c2,r1,r2,N,D,M参数初始化
    c1 = c2 = 1.55
    N = 20
    D = 2
    M = 1000
    pso_object = PSO_model(c1, c2, N, D, M)  # 设置初始权值
    pso_object.init_pop()
    pso_object.update()
    pso_object.loop()
