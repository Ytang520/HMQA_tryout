import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model

class Lstm_regression(Model):

    '''
    Lstm_regression
    输入：input
    输出：
        out：regression输出 (需考虑不同时间步不同的loss权值
        # 注意此处 linear 输出以保证输出稳定 (sigmoid比较难控制得分在0~1之间的分数)
    '''

    def __init__(self):

        super(Lstm_regression, self).__init__()

        # Temporal_pyramid
        self.temporal_pyramid_1 = tf.keras.layers.Conv2D(4, (1, 1), padding='same', activation='relu')
        self.temporal_pyramid_2 = tf.keras.layers.Conv2D(4, (9, 1), padding='same', activation='relu')
        self.temporal_pyramid_3 = tf.keras.layers.Conv2D(4, (15, 1), padding='same', activation='relu')
        self.temporal_pyramid_4 = tf.keras.layers.Conv2D(4, (20, 1), padding='same', activation='relu')

        # LSTM block
        self.lstm_1 = LSTM(80, return_sequences=True)
        self.lstm_2 = LSTM(40, return_sequences=True)
        self.lstm_3 = LSTM(40, return_sequences=True)
        self.lstm_4 = LSTM(80)
        self.out_dense = Dense(1)


    def call(self, inter_inputs):

        """Temporal convolution"""
        # 扩张维度，并构建时间金字塔
        z = tf.expand_dims(inter_inputs, axis=3)
        z1 = self.temporal_pyramid_1(z)
        z1 = Dropout(0.25)(z1)
        z2 = self.temporal_pyramid_2(z1)
        z2 = Dropout(0.25)(z2)
        z3 = self.temporal_pyramid_3(z2)
        z3 = Dropout(0.25)(z3)
        z4 = self.temporal_pyramid_4(z3)
        z4 = Dropout(0.25)(z4)
        pyramid_out = concatenate([z1, z2, z3, z4], axis=-1)

        # LSTM 时间信息捕捉
        x = self.lstm_1(pyramid_out)
        x = Dropout(0.25)(x)
        x = self.lstm_2(x)
        x = Dropout(0.25)(x)
        x = self.lstm_3(x)
        x = Dropout(0.25)(x)
        x = self.lstm_4(x)
        out = self.out_dense(x)

        return out