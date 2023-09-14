import numpy as np
import matplotlib.pyplot as plt

import scipy.signal as signal
from scipy.signal import find_peaks, butter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import os
import pandas as pd
def rcsv_wo_fullname(path, prefix='JointPosition', header=None):
    '''
    仅根据前缀读取csv文件
    输入参数：
        path: 表示存储文件的文件夹路径
        prefix='JointPosition': 表示带读取的文件前缀，此处默认为 'JointPosition'
    返回：
        df: pandas读入的 csv文件
    '''
    files = os.listdir(path)
    selected_files = [f for f in files if f.startswith(prefix)]
    for filename in selected_files:
        filepath = os.path.join(path, filename)  # 拼接文件路径
        df = pd.read_csv(filepath, header=None)  # 读取 CSV 文件
        df.dropna(axis=1, how='all', inplace=True)
    return df

path = 'D:\Desktop\dataset\KiMoRe\CG\Expert\E_ID2\Es5\Raw'
df = rcsv_wo_fullname(path, header=None)  # 注意此处会去掉了前面数行空行
x = df.iloc[:, 70].values  # 表现较好
x = np.reshape(x, (-1, 1))
x = StandardScaler().fit_transform(x)  # 不进行归一化，归一化之后效果表现很差
x = np.squeeze(x, axis=-1)
t = np.arange(x.shape[0])

filtCutOff = 1
sample=30
b, a = butter(3, (2*filtCutOff)/sample, 'lowpass')  # 3阶低通滤波器，
y = signal.filtfilt(b, a, x)  # y表示已经低通滤波后的结果

# 寻找峰值
peaks, _ = find_peaks(y)
plt.plot(t[peaks], x[peaks], 'x')
plt.xlabel('Time (s)')
plt.ylabel('Signal Amplitude')
plt.title('Peak Detection Example')
plt.show()

fig, axs = plt.subplots(2)
fig.suptitle('Lowpass Filter Example')
axs[0].plot(t, x)
axs[0].set_title('Original Signal')
axs[1].plot(t, y)
axs[1].set_title('Filtered Signal')
plt.show()


