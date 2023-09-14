import math

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class Self_Attention(layers.Layer):
    '''
    Transformer 中 self-attention 操作，得到对于每一个head下的结果
    参数：
        inputs: [None, T, num]
        size_to_hidden: 原始输入层的feature大小num/隐藏层的数量的feature大小
        size: 节点个数, inputs[-1]
    输入：
        [None, T, num]
    输出：
        [None, T, num]
    '''

    def __init__(self, size, size_to_hidden=4):
        super(Self_Attention, self).__init__()
        # inputs: [None, T, num]

        self.size_to_hidden = size_to_hidden
        self.W_q = tf.Variable(lambda: tf.random.normal(shape=[size, size // size_to_hidden]), dtype=tf.float32,
                               trainable=True)
        self.W_k = tf.Variable(lambda: tf.random.normal(shape=[size, size // size_to_hidden], dtype=tf.float32),
                               trainable=True)
        self.W_v = tf.Variable(lambda: tf.random.normal(shape=[size, size // size_to_hidden]), dtype=tf.float32,
                               trainable=True)

    def call(self, inputs):
        # [None, T, num]
        Q = tf.einsum("ijk,kl->ijl", inputs, self.W_q)
        K = tf.einsum("ijk,kl->ijl", inputs, self.W_k)

        # 计算并返回缩放因子  [None, T, T]
        alpha = tf.einsum("ijk,ikl->ijl ", Q, tf.transpose(K, [0, 2, 1]))
        alpha = tf.nn.softmax(alpha / tf.math.sqrt(float(self.size_to_hidden)), axis=-1)

        # 得到相应的结果
        V = tf.einsum("ijk,kl->ijl", inputs, self.W_v)
        results = tf.einsum("ijk,ikl->ijl", alpha, V)

        return results


class Coder(layers.Layer):
    '''
    Transformer的coder, 编码和解码目标；值得注意的是，有些可训练变量是在 call 里定义的，原因在于其会随着问题的解法不同而发生变化
    参数：
        size: 节点个数, inputs[-1]
        head_num=8：头个数
        size_to_hidden=4：原始输入层的feature大小num/隐藏层的数量的feature大小
        epsilon: normalization 时参数
    输入：
        [None, T, num]
    输出：
        [None, T, num]
    '''

    def __init__(self, size, head_num=8, size_to_hidden=4, epsilon=1e-6):
        super(Coder, self).__init__()

        self.head_num = head_num
        self.multi_head = [Self_Attention(size=size, size_to_hidden=size_to_hidden) for i in range(self.head_num)]
        self.epsilon = epsilon

        # W_o 建立
        self.W_o = tf.Variable(
            lambda: tf.random.normal(shape=[size // size_to_hidden * head_num, size], dtype=tf.float32),
            trainable=True)  # W_o shape =[size/size_to_hidden*multi_head, input_num]

        # add_norm 参数设计
        self.gamma = tf.Variable(lambda: tf.ones(shape=(1,), dtype=tf.float32), trainable=True)
        self.beta = tf.Variable(lambda: tf.zeros(shape=(1,), dtype=tf.float32), trainable=True)

        # feed_forward 设计
        self.Dense_1_1 = layers.Dense(units=size * 2, activation=tf.nn.leaky_relu)

        # add_norm_2 参数设计
        self.gamma_2 = tf.Variable(lambda: tf.ones(shape=(1,), dtype=tf.float32), trainable=True)
        self.beta_2 = tf.Variable(lambda: tf.zeros(shape=(1,), dtype=tf.float32), trainable=True)

        # feed_forward_2 设计
        self.Dense_1_2 = layers.Dense(units=size)

    def call(self, inputs):
        # encoder
        out_multihead = []
        for i in range(self.head_num):
            out_multihead.append(self.multi_head[i](inputs))

        # out_multihead = [self.multi_head[i](inputs) for i in range(self.head_num)]  # 自定义头的数量, 并作乘法
        encoder_output = tf.concat(out_multihead, axis=-1)  # shape=[None, T, size/size_to_hidden*multi_head]
        outputs = tf.einsum("ijk,kl->ijl", encoder_output, self.W_o)  # 与输入shape一致，[None, T, num]

        # add_norm
        outputs = tf.add(inputs, outputs)  # add
        mean, variance = tf.nn.moments(outputs, axes=[-1], keepdims=True)  # norm开始, 这里注意轴是axes
        normalized_inputs = tf.math.divide_no_nan((inputs - mean), tf.math.sqrt(variance + self.epsilon))
        outputs = self.gamma * normalized_inputs + self.beta

        inputs = outputs  # 方便后续处理

        # feed_forward
        out = self.Dense_1_1(outputs)
        out = tf.nn.dropout(out, rate=0.2)  # 部分置0，而其余rescaled x/(1-rate), 此时总体输入一样
        out = self.Dense_1_2(out)
        outputs = tf.nn.dropout(out, rate=0.2)  # 部分置0，而其余rescaled x/(1-rate), 此时总体输入一样

        # add_norm_2
        outputs = tf.add(inputs, outputs)  # add
        mean, variance = tf.nn.moments(outputs, axes=[-1], keepdims=True)  # norm开始
        normalized_inputs = tf.math.divide_no_nan((inputs - mean), tf.math.sqrt(variance + self.epsilon))
        outputs = self.gamma_2 * normalized_inputs + self.beta_2

        return outputs


class Transformer(layers.Layer):
    '''
    输入：
        shape = [None, T, N, C_in]
    参数：
        num: inputs.shape[-1]*inputs.shape[-2]  num = N*C_in
        head_num=8：表示头的数目
        hidden_feature=8：表示encoder输出层的feature大小(num)
        input_feature=16：表示encoder输入时的feature大小(num)
        size_to_hidden=4：原始输入层的feature大小num/隐藏层的数量的feature大小
        epsilon=1e-6: normalization 时参数
        type='AE': 表示是否使用AE结构，可选输入为 'AE' 和 ‘Regression’
    输出：
        outputs: [None, T, N * initial_in_channel]
        encoder_outputs: shape = [None, T, initial_in_channel]
    注意：
        部分可训练变量在call中定义，原因在于其形式会随着输入的不同而发生变化

    '''

    def __init__(self, encoder_input_size=300, channels=3, head_num=8, hidden_feature=8,  size_to_hidden=4, epsilon=1e-6, type='AE'):
        super(Transformer, self).__init__()
        self.head_num = head_num
        self.channels = channels
        self.hidden_feature = hidden_feature  # 此处未用到，原因在于没有显式的使用瓶颈层结构
        self.encoder_input_size = encoder_input_size  # encoder_input_size = 25(Nodes)*48(GCN_channel_output)=1200

        if type == 'AE':
            self.type_choose = True
            self.Conv1D_3 = layers.Conv1D(filters=self.channels * 25 * 2, kernel_size=1, strides=1, padding='valid',
                                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                          bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                          trainable=True, name='test')
            self.Conv1D_4 = layers.Conv1D(filters=self.channels * 25, kernel_size=1, strides=1, padding='valid',
                                          kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                          bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                          trainable=True)

        else:
            self.type_choose = False
            self.output_1 = layers.Conv1D(filters=30, kernel_size=5, strides=5, padding='valid',
                                      kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                      bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                      trainable=True, name='output_1')
            self.dropout_1 = layers.Dropout(0.25)
            self.output_2 = layers.Conv1D(filters=30, kernel_size=5, strides=1, padding='valid',
                                      kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                      bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                      trainable=True, name='output_2')
            self.dropout_2 = layers.Dropout(0.25)
            self.flatten = layers.Flatten()
            # self.output_3 = layers.Dense(units=1, activation=tf.nn.sigmoid, trainable=True, name='Dense')
            self.output_3 = layers.Dense(units=1, activation='linear', trainable=True, name='Dense')  # 尝试 linear 作结尾

        # connection_agcn_trans 参数定义
        self.Conv1D_1 = layers.Conv1D(filters=math.ceil(self.encoder_input_size/4), kernel_size=1, strides=1, padding='valid',
                                      kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                      bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                      trainable=True)
        self.Conv1D_2 = layers.Conv1D(filters=30, kernel_size=1, strides=1, padding='valid',
                                      kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                      bias_initializer=tf.keras.initializers.Zeros(), activation=tf.nn.leaky_relu,
                                      trainable=True)

        # encoder & decoder 定义
        self.encoder = Coder(30, head_num=head_num, size_to_hidden=size_to_hidden, epsilon=epsilon)
        #  encoder_out_channel = math.ceil(self.encoder_input_size/40)
        self.decoder = Coder(30, head_num=head_num, size_to_hidden=size_to_hidden, epsilon=epsilon)

        # 重回到原来的维度

    def positional_embedding(self, max_len):
        '''
        此方法实现序列的顺序编码压缩，其中待嵌入向量的维度直接定义为 self.encoder_input_size, 不再设置参数
        参数：
            max_len: 输入序列最大长度
        输出：
            [T,num]
        '''
        pos = np.arange(max_len)[:, np.newaxis]  # 在第二维度增加新的轴
        i = np.arange(30)[np.newaxis, :]  # 在第一维(最外面)增加新的轴
        angle = pos / (10000 ** (2 * i / 30))
        sin = np.sin(angle[:, 0::2])
        cos = np.cos(angle[:, 1::2])
        pos_encoding = np.concatenate([sin, cos], axis=-1)
        pos_encoding = tf.convert_to_tensor(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def connection_agcn_trans(self, inputs):
        '''
        连接AGCN和Transformer，并嵌入embedding
        输入：
            [None, T, N, C_in]
        输出：
            [None, T, f(encoder_input_size)]  # f(encoder_input_size)表示对encoder_input_size大小做出调整后的结果, 此处设定为30
        '''
        # [None, T, N, C_in] -> [None, T, num]
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        inputs = tf.reshape(inputs, (-1, inputs.shape[1], inputs.shape[2] * inputs.shape[3]))
        # print(inputs.shape)

        # 一维卷积，得到trans的输入
        out = self.Conv1D_1(inputs)
        out = self.Conv1D_2(out)  # shape: [None, T, self.encoder_input_size]

        # 顺序 embedding 嵌入
        embedding = self.positional_embedding(inputs.shape[1])  # inputs.shape[1]为T
        embedding_out = out + embedding  # 可让其自动广播相加，无需人为指定

        return embedding_out

    def call(self, inputs):
        '''
        实现整体的网络连接
        输入：
            [None, T, N, C_in]
        输出：
            Regression:
            AE:[None, T, N*C_in]
        '''
        inputs = self.connection_agcn_trans(inputs)  # 由AGCN的输出shape变为Transformer的输入shape
        encoder_out = self.encoder(inputs)
        out = self.decoder(encoder_out)  # out: shape=[None, T, num]

        # !!! 此处暂未实现
        # 降维, 是否一定要实现？可以直接借助encoder实现了（虽然不标准）
        # !!!

        if self.type_choose:
            # 返回原有的维度
            out = self.Conv1D_3(out)
            out = self.Conv1D_4(out)
            out = tf.reshape(out, (-1, out.shape[1], 25, 3))
            return out, encoder_out

        else:
            out = self.output_1(out)
            out = self.dropout_1(out)
            out = self.output_2(out)
            out = self.dropout_2(out)
            out = self.flatten(out)
            # print('out_shape', out.shape)
            out = self.output_3(out)
            return out

        # 多头hidden_feature, 并且 linear层 得到最终multi-head attention
