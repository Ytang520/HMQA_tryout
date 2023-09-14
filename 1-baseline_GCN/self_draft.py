# import numpy as np
# from scipy.signal import argrelextrema
# import matplotlib.pyplot as plt
# from scipy import signal
#
#
# # 定义一段波形信号
# t = np.linspace(0, 1, 1000)
# y = np.sin(2 * np.pi * 5 * t) + 0.2 * np.sin(2 * np.pi * 50 * t)
#
# filtCutOff = 1
# order = 3
# sample = 30
#
# # 创建低通滤波器
# nyquist_freq = 0.5 * sample
# cutoff_norm = filtCutOff / nyquist_freq
# b, a = signal.butter(order, cutoff_norm, btype='low')
#
# # 使用前向和反向滤波来消除相位延迟
# y = signal.filtfilt(b, a, y)
#
# # 使用 argrelextrema 函数寻找极小值点
# minima_indices = argrelextrema(y, np.less)
#
# # 输出极小值点的索引和值
# minima_values = y[minima_indices]
# print("Minima indices: ", minima_indices)
# print("Minima values: ", minima_values)
#
# # 绘制信号和极小值点
# plt.plot(t, y, label='Signal')
# plt.plot(t[minima_indices], y[minima_indices], 'o', label='Minima')
# plt.legend()
# plt.show()

# import nu

# import numpy as np
# import matplotlib.pyplot as plt
#
# x = np.random.normal(size=(2,3))
# print(x)
# print(x[:, 0])

# 定义原始数据, 不是终末两个值进行插值，而是每个值都会计算进行线性插值
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([0, 2, 1, 3, 11, 11])
#
# # 定义插值点
# x_interp = np.linspace(1, 4, 11)
#
# # 进行线性插值
# y_interp = np.interp(x_interp, x, y)
#
# # 输出插值结果
# print('x_interp:', x_interp)
# print('y_interp:', y_interp)
# print(len(y_interp))
# plt.plot(y_interp, label='check')
# plt.legend()
# plt.show()


# 测试当函数参数是网络时,传入的是副本还是地址: 结果表明是地址
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential


def modify_network(my_network):
    my_network.add(Dense(64, activation='relu'))

# 创建一个神经网络
network1 = Sequential([
    Dense(32, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

# 调用函数修改神经网络
modify_network(network1)

# 打印神经网络的结构
network1.summary()