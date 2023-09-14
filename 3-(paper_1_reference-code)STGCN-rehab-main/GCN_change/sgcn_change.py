import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, Input, LSTM, concatenate, ConvLSTM2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *
# from IPython.core.debugger import set_trace
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers


class gcn(layers.Layer):
    '''
    gcn_module
    '''
    def __init__(self, channels=64):
        super(gcn, self).__init__()
        self.Conv2D = tf.keras.layers.Conv2D(filters=channels, kernel_size=(1, 1), strides=1, activation='relu')
        self.Convlstm = ConvLSTM2D(filters=25, kernel_size=(1, 1), input_shape=(None, None, 25, 1, 3), return_sequences=True)
        self.bias_mat = tf.Variable(tf.ones((25, 25)) * 0.01)

    def call(self, Input):
        x1 = self.Conv2D(Input)
        expand_x1 = tf.expand_dims(x1, axis=3)
        f_1 = self.Convlstm(expand_x1)
        f_1 = f_1[:, :, :, 0, :]
        logits = f_1
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + self.bias_mat)


        gcn_x1 = tf.keras.layers.Lambda(lambda x: tf.einsum('ntvw,ntwc->ntvc', x[0], x[1]))([coefs, x1])

        # 这里最好还是sparse attention，表示对于不同任务有不同的 attention 要求

        return gcn_x1


class Sgcn_Lstm():
    def __init__(self, train_x, train_y, AD, AD2, bias_mat_1, bias_mat_2, lr=0.0001, epoch=200, batch_size=10):
        self.train_x = train_x
        self.train_y = train_y
        self.AD = AD
        self.AD2 = AD2
        self.bias_mat_1 = bias_mat_1
        self.bias_mat_2 = bias_mat_2
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_joints = 25

        # gcn 和 sparse 模块
        self.T_conv = tf.keras.layers.Conv2D(64, (9, 1), padding='same', activation='relu')
        self.gcn_1 = gcn(channels=64)
        self.gcn_2 = gcn(channels=64)
        self.Tempy_1_conv = tf.keras.layers.Conv2D(16, (9, 1), padding='same', activation='relu')
        self.Tempy_2_conv = tf.keras.layers.Conv2D(16, (15, 1), padding='same', activation='relu')
        self.Tempy_3_conv = tf.keras.layers.Conv2D(16, (20, 1), padding='same', activation='relu')

        # Lstm_AE模块
        self.lstm_AE_convlstm = ConvLSTM2D(filters=12, kernel_size=(1, 1), strides=1, activation='relu', return_sequences=True)
        self.lstm_AE_lstm = LSTM(filters=75, return_sequences=True)

    def sgcn(self, Input):
        """Temporal convolution"""
        k1 = self.T_conv(Input)
        k = concatenate([Input, k1], axis=-1)
        """Graph Convolution"""

        """ 一二阶构造(看是否需要通过一阶然后乘积实现？) """
        gcn_x1 = self.gcn_1(k)
        gcn_y1 = self.gcn_2(k)
        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

        """Temporal convolution"""
        z1 = self.Tempy_1_conv(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = self.Tempy_2_conv(z1)
        z2 = Dropout(0.25)(z2)
        z3 = self.Tempy_3_conv(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1)

        return z

    def sgcn_sparse(self, Input):

        """Temporal convolution"""
        k1 = self.T_conv(Input)
        k = concatenate([Input, k1], axis=-1)
        """Graph Convolution"""

        """ 一二阶构造(看是否需要通过一阶然后乘积实现？) """
        gcn_x1 = self.gcn_1(k)
        gcn_y1 = self.gcn_2(k)
        gcn_1 = concatenate([gcn_x1, gcn_y1], axis=-1)

        # 加入sparsity (对每一个时刻 T 处理，每张图里都是sprase)

        """Temporal convolution"""
        z1 = self.Tempy_1_conv(gcn_1)
        z1 = Dropout(0.25)(z1)
        z2 = self.Tempy_2_conv(z1)
        z2 = Dropout(0.25)(z2)
        z3 = self.Tempy_3_conv(z2)
        z3 = Dropout(0.25)(z3)
        z = concatenate([z1, z2, z3], axis=-1)

        return z


    def Lstm(self, x):
        x = tf.keras.layers.Reshape(target_shape=(-1, x.shape[2] * x.shape[3]))(x)
        x = LSTM(80, return_sequences=True)(x)
        x = Dropout(0.25)(x)
        x = LSTM(40)(x)
        x = Dropout(0.25)(x)

        # rec = LSTM(80, return_sequences=True)(x)
        # rec = Dropout(0.25)(rec)
        # rec1 = LSTM(40, return_sequences=True)(rec)
        # rec1 = Dropout(0.25)(rec1)
        # rec2 = LSTM(40, return_sequences=True)(rec1)
        # rec2 = Dropout(0.25)(rec2)

        out = Dense(1, activation='linear')(x)
        return out

    def Lstm_AE(self, x):
        '''
        x_shape = [N, 100, 25, 48]
        使用ConvLSTM2D 和 lstm, 得到AE的结果
            out_shape = [N, 100, 25, 3]
        '''

        x = tf.expand_dims(x, axis=3)
        x = self.lstm_AE_convlstm(x)
        x = tf.squeeze(x)
        x = Dropout(0.25)(x)
        x = tf.reshape(x, (-1, 100, x.shape[-1]*x.shape[-2]))
        x = self.lstm_AE_lstm(x)
        out = tf.reshape(x, (-1, 100, 25, 3))

        return out

    def train_AE(self):
        '''
            训练AE, 并且得到结果
        '''
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        # 得到多阶图(空间相关性)
        x = self.sgcn_sparse(seq_input)
        y = self.sgcn_sparse(x)
        y = y + x
        z = self.sgcn_sparse(y)
        z = z + y

        out = self.Lstm_AE(z)
        self.model_AE = Model(seq_input, out)
        self.model_AE.compile(loss=tf.keras.losses.Huber(delta=0.1), steps_per_execution=50,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics=['mae', 'mse'])
        history = self.model_AE.fit(self.train_x, self.train_x, validation_split=0.2, epochs=self.epoch,
                                 batch_size=self.batch_size)

        return history


    def train(self):
        seq_input = Input(shape=(None, self.train_x.shape[2], self.train_x.shape[3]), batch_size=None)
        # 得到多阶图(空间相关性)
        x = self.sgcn(seq_input)
        y = self.sgcn(x)
        y = y + x
        z = self.sgcn(y)
        z = z + y

        # 加入 sparsity ？

        out = self.Lstm(z)
        self.model = Model(seq_input, out)
        self.model.compile(loss=tf.keras.losses.Huber(delta=0.1), steps_per_execution=50,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr), metrics=['mae', 'mape'])
        checkpoint = ModelCheckpoint("best model ex4/best_model.hdf5", monitor='val_loss', save_best_only=True,
                                     mode='auto', save_freq=1)
        self.model.summary()
        history = self.model.fit(self.train_x, self.train_y, validation_split=0.2, epochs=self.epoch,
                                 batch_size=self.batch_size, callbacks=[checkpoint])
        return history

    def prediction(self, data):
        y_pred = self.model.predict(data)
        return y_pred


class siamese_LSTM():
    pass