import tensorflow as tf
from AGCN import *

class LSTM_part(tf.keras.layers.Layer):
    def __init__(self):
        super(LSTM_part, self).__init__()
        self.seq = tf.keras.Sequential()
        self.seq.add(layers.LSTM(units=80, return_sequences=True))
        self.seq.add(layers.Dropout(0.25))
        self.seq.add(layers.LSTM(units=40, return_sequences=True))
        self.seq.add(layers.Dropout(0.25))
        self.seq.add(layers.LSTM(units=40, return_sequences=True))
        self.seq.add(layers.Dropout(0.25))
        self.seq.add(layers.LSTM(units=80))
        self.seq.add(layers.Dropout(0.25))
        self.seq.add(layers.Dense(1, activation='linear'))

        # self.LSTM_1 = layers.LSTM(units=80, return_sequences=True)
        # self.Dropout_1 = layers.Dropout(0.25)
        # self.LSTM_2 = layers.LSTM(units=40, return_sequences=True)
        # self.Dropout_2 = layers.Dropout(0.25)
        # self.LSTM_3 = layers.LSTM(units=40, return_sequences=True)
        # self.Dropout_3 = layers.Dropout(0.25)
        # self.LSTM_4 = layers.LSTM(units=80)
        # self.Dropout_4 = layers.Dropout(0.25)
        # self.Dense = layers.Dense(1, activation='linear')

    def call(self, inputs):
        out = self.seq(inputs)

        return out


class AGCN_LSTM(tf.keras.Model):
    '''
        AGCN和LSTM模块拼接
    参数：
        matrix：原始邻接图
        matrix_2=None: 二阶邻接图, 为可选项
        type='AE': 表示是否使用AE结构，可选输入为 'AE' 和 ‘Regression’
    输入：
        inputs: shape = [None, T, N, C_in]
    输出：
        out : shape = [1]


    '''
    def __init__(self, matrix, matrix_2=None):
        super(AGCN_LSTM, self).__init__()
        # 此处未进行siamese操作
        self.AGCN = AGCN_TCN(matrix, matrix_2=matrix_2)
        self.LSTM_part = LSTM_part()


    def call(self, inputs):
        out = self.AGCN(inputs)
        out = tf.reshape(out, [-1, out.shape[1], out.shape[2] * out.shape[3]])
        out = self.LSTM_part(out)

        return out
