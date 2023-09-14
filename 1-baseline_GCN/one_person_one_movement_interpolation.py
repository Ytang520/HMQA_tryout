import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def data_filtering(path, whether_filter=True):
    '''
    1. 对原始数据进行格式转换
    2. 在原始数据未经过处理(“污染”)时，先进行标准化
    3. 根据需要对标准化后的数据进行滤波
    输入：
        path
        whether_filter：是否滤波，默认值为True
    输出：
        train_x : 标准化后的数据, array数据结构
        filter_train_x：表示滤波后的数据，若whether_filter=False, 则为标准化后的数据, array数据结构
    '''

    train_x = pd.read_csv(path, header=None, dtype=np.float64)  # 应该不必iloc
    col_drop = [4 * i + 3 for i in range(int(train_x.shape[1] / 4))]
    train_x.drop(train_x.columns[col_drop], axis=1, inplace=True)    # shape = [求和 N_i*seq_len_i, 25*3]

    # 对数据标准化
    train_x = StandardScaler().fit_transform(train_x)

    # 参照最开始的HMQA文章的matlab上做法，使用低通滤波器；对每个人的所有动作，每个关节的每个坐标进行滤波
    filter_train_x = train_x
    if whether_filter:
        filtCutOff = 1
        order = 3
        sample = 30

        # 创建低通滤波器
        nyquist_freq = 0.5 * sample
        cutoff_norm = filtCutOff / nyquist_freq
        b, a = signal.butter(order, cutoff_norm, btype='low')

        # 使用前向和反向滤波来消除相位延迟
        filter_train_x = signal.filtfilt(b, a, train_x, axis=0)

    # 每个动作具有特殊性，找到最小值后需要人为进行划分 (或许需要删减之类的
    return train_x, filter_train_x

def data_segementation(data, joint_axis_selection=None):
    '''
    数据分割，得到分割数据位置列表pos_list (pos不含末尾位置
    输入：
        data
        joint_axis_selection: 不同动作所选择的关节 (人为观察选择，此处尽量选择易于进行划分的关节坐标, 即初始为最小值，而后两个最小值之间为一次动作
             —— 一般比较困难
    输出：
        pos_list: 需要切分的位置(不包含尾部的位置), list数据结构
    '''

    pass


def data_interpolation(segmented_data, pos_list, per_time_step=100):
    '''
    线性插值得到目标时间步长的数据
    输入：
        segmented_data: array数据结构
        pos_list: 需要切分的位置(不包含尾部的位置), list数据结构
        per_time_step: 此处处理的时间步(可以任意长度，方便后续variational length的说明), 默认值为100
    输出：
        interpolated_data
        shape:[time_step=100, 75, num_seq]
    '''

    pos_list.append(segmented_data.shape[0]-1)  # 使得需要切分的位置包含最后一位

    interpolated_data = []
    for i in pos_list:
        if i != pos_list[-1]:
            x = np.arange(pos_list[i], pos_list[i+1])
            x_interp = np.linspace(pos_list[i], pos_list[i+1], per_time_step)
            data_interp = []
            for j in range(75):  # 对每列进行插值
                y_interp = np.interp(x_interp, x, segmented_data[:, j])
                data_interp.append(y_interp)
            data_interp = np.array(data_interp).T  # 直接求二维矩阵转置
            interpolated_data.append(data_interp)
        else:
            break
    interpolated_data = np.array(interpolated_data)  # shape:[100, 75, num_seq]

    return interpolated_data