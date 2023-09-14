import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D, Conv2D
from tensorflow.keras.models import Model



class GCN_AE(Model):

    '''
    GCN_AE
    输入：input
    输出：
        inter_output：中间层 (需加入sparsity), shape = [N, 100, 25*2]
        out：AE输出 (需考虑不同时间步不同的loss权值
    '''

    def __init__(self):

        super(GCN_AE, self).__init__()

        # for_GCN
        self.conv_T = Conv2D(64, (9, 1), padding='same', activation='relu')
        self.hop_1_GCN = Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation='relu')
        self.convlstm_1_GCN = ConvLSTM2D(filters=25, kernel_size=(1, 1), input_shape=(None, None, 25, 1, 3), return_sequences=True)
        self.hop_2_GCN = Conv2D(filters=64, kernel_size=(1, 1), strides=1, activation='relu')
        self.convlstm_2_GCN = ConvLSTM2D(filters=25, kernel_size=(1, 1), input_shape=(None, None, 25, 1, 3), return_sequences=True)
        # out shape = [N, 100, 25, 128]

        # for_AE_features
        self.reweight_gcn_out = Conv2D(64, (1, 1), padding='same', activation='relu')  # [N, 100, 25, 128] -> [N, 100, 25, 64]
        # 此处假定自由度的捕捉与时间有关
        self.convlstm_1 = ConvLSTM2D(32, (1, 1), padding='same', activation='relu', return_sequences=True)  # [N, 100, 25, 1, 64] -> [N, 100, 25, 1, 32]
        self.convlstm_2 = ConvLSTM2D(16, (1, 1), padding='same', activation='relu', return_sequences=True)  # [N, 100, 25, 1, 32] -> [N, 100, 25, 1, 16]
        self.convlstm_3 = ConvLSTM2D(8, (1, 1), padding='same', activation='relu', return_sequences=True)  # [N, 100, 25, 1, 16] -> [N, 100, 25, 1, 8]
        # [N, 100, 25, 1, 8] -> [N, 100, 25*8]
        self.lstm_4 = LSTM(100, return_sequences=True)  # [N, 100, 25*8] -> [N, 100, 25*4]
        self.lstm_5 = LSTM(50, return_sequences=True)  # [N, 100, 25*4] -> [N, 100, 25*2]

        # for_AE_reconstrution_features
        self.rec_lstm_1 = LSTM(150, return_sequences=True)  # [N, 100, 25*2] -> [N, 100, 25*6]
        self.rec_lstm_2 = LSTM(75, return_sequences=True)  # [N, 100, 25*6] -> [N, 100, 25*3]



    def call(self, Inputs):

        # GCN_part

        # """Temporal convolution"""
        k1 = self.conv_T(Inputs)
        k = concatenate([Inputs, k1], axis=-1)

        # """Graph Convolution"""
        # """first hop localization"""
        x1 = self.hop_1_GCN(k)
        expand_x1 = tf.expand_dims(x1, axis=3)
        f_1 = self.convlstm_1_GCN(expand_x1)
        logits = f_1[:, :, :, 0, :]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_1)  # coef 是 attention map
        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])

        # """second hop localization"""
        y1 = self.hop_2_GCN(k)
        expand_y1 = tf.expand_dims(y1, axis=3)
        f_2 =self.convlstm_2_GCN(expand_y1)
        logits = f_2[:, :, :, 0, :]
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat_2)
        gcn_y1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, y1])

        # reweight gcn_out
        gcn_out = concatenate([gcn_x1, gcn_y1], axis=-1)
        gcn_out = self.reweight_gcn_out(gcn_out)

        # 填充扩张后，得到inter_output
        gcn_out = tf.expand_dims(gcn_out, axis=3)
        x = self.convlstm_1(gcn_out)
        x = self.convlstm_2(x)
        x = self.convlstm_3(x)
        x = x[:, :, :, 0, :]
        x = tf.reshape(x, (-1, 100, 25*8))
        x = self.lstm_4(x)
        inter_output = self.lstm_5(x)

        # 填充扩张后，得到inter_output
        x = self.rec_lstm_1(inter_output)
        x = self.rec_lstm_2(x)
        out = tf.reshape(x, (-1, 100, 25, 3))

        return inter_output, out