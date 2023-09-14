import tensorflow as tf
from tensorflow.keras import layers
import math

class AGCN_attention_for_LSTM(layers.Layer):
    '''
            AGCN_TCN_block 搭建, 参数包含 out_channels, A, coff_embedding=4, kernel_size=9, stride=1
            初始化参数：
                out_channels：表示输出 channel 的维度
                A：邻接图, 需要为 tensor 类型
                coff_embedding：压缩的维度
                kernel_size：TCN 中对时间维度做卷积时，其卷积核大小，默认为9，scalar值
                stride：TCN 中对时间维度做卷积时，其补偿大小，默认为1，scalar值
            输入：
                inputs_block: shape = [None, T, N, C_in]
            输出：
                output_AGCN_TCN：shape = [None, T, N, C_out]
        '''

    def __init__(self, out_channels, A, coff_embedding=12):
        super(AGCN_attention_for_LSTM, self).__init__()
        self.matrix = A
        self.out_channels = out_channels
        self.convlstm = layers.ConvLSTM2D(filters=A.shape[0], kernel_size=(1, 1), padding="same",
                                          return_sequences=True, data_format='channels_last')
        # emmmm, 没有ConvLSTM1D模块

        self.Conv2D_output = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=[1, 1],
            strides=[1, 1],
            padding='valid',
            activation=tf.nn.leaky_relu,
            trainable=True)

    def call(self, inputs_per_part):
        # attention_graph = convlstm_graph  # 目前先不加入自定义的matrix

        inputs_per_part_raw = tf.expand_dims(inputs_per_part, axis=3)
        convlstm_graph = self.convlstm(inputs_per_part_raw)  # shape=(None, T, N, 1, N)
        convlstm_graph = tf.squeeze(convlstm_graph)

        # print(convlstm_graph.shape)

        inputs_per_part_raw = tf.transpose(inputs_per_part, [0, 1, 3, 2])  # shape=(None, T, C_e, N)
        output_middle = tf.einsum('ijkw,ijwc->ijkc', inputs_per_part_raw, convlstm_graph)  # 这里为四维，原因在于None：batchsize
        # 此处output_middle为: shape=(None, C_e, T, N)

        output_middle = tf.transpose(output_middle, [0, 1, 3, 2])  # shape = [None, T, N, C_e]
        output = self.Conv2D_output(output_middle)

        return output



class TCN_part(layers.Layer):
    '''
        TCN_part 搭建, 参数包含 out_channels, A, coff_embedding=4, kernel_size=9, stride=1
        初始化参数：
            out_channels：表示输出 channel 的维度
            kernel_size：TCN 中对时间维度做卷积时，其卷积核大小，默认为9，scalar值
            stride：TCN 中对时间维度做卷积时，其补偿大小，默认为1，scalar值
        输入：
            inputs_block: shape = [None, T, N, C_in]
        输出：
            output_AGCN_TCN：shape = [None, T, N, C_out]
    '''

    def __init__(self, out_channels, kernel_size=10, stride=1):
        super(TCN_part, self).__init__()
        self.Conv2D_TCN = layers.Conv2D(
            filters=out_channels,
            kernel_size=(kernel_size, 1),  # kernel_size 第一个维度为height，第二给维度为 width
            strides=(stride, 1),
            padding='same',  # 这里进行填充操作, 使得和input相同的结构
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.keras.initializers.Zeros(),
            activation=tf.nn.leaky_relu,
        )

        # 加速计算：Batchnorm和relu
        self.Batchnorm = layers.BatchNormalization()  # 对每一层都进行batchnormalization，加速网络; 这里不必加入任何参数
        # layers.Flatten() 多维展开成一维, 用于后续regression
        self.relu = layers.LeakyReLU()  # 对每一个元素都进行 ReLU 操作，让输出更稳定

    def call(self, inputs_per_part):
        output = self.Conv2D_TCN(inputs_per_part)  # shape = [None, T, N, C_out]
        output_batchnorm = self.Batchnorm(output)
        output = self.relu(output_batchnorm)
        # print('/', output.shape)
        return output


class AGCN_TCN_block(layers.Layer):
    '''
        AGCN_TCN_block 搭建, 参数包含 out_channels, A, coff_embedding=4, kernel_size=9, stride=1
        初始化参数：
            out_channels：表示输出 channel 的维度
            A：邻接图
            coff_embedding：压缩的维度
            kernel_size：TCN 中对时间维度做卷积时，其卷积核大小，默认为9，scalar值
            stride：TCN 中对时间维度做卷积时，其补偿大小，默认为1，scalar值
        输入：
            inputs_block: shape = [None, T, N, C_in]
        输出：
            output_AGCN_TCN：shape = [None, T, N, C_out]
    '''

    def __init__(self, out_channels, A, coff_embedding=12, kernel_size=10, stride=1):
        super(AGCN_TCN_block, self).__init__()

        # 此处使用 convlstm 模块的attention机制
        # self.AGCN_part = AGCN_part(out_channels, A)
        self.AGCN_part = AGCN_attention_for_LSTM(out_channels, A)

        self.TCN_part = TCN_part(out_channels, kernel_size, stride)
        self.dropout = layers.Dropout(rate=0.5)
        self.stride = stride
        self.TCN_for_add_norm = TCN_part(out_channels, kernel_size=1, stride=1)

    def call(self, inputs_per_layer):
        # AGCN_block 搭建, 这里有问题，A, B, C 不是两者，不过此处暂且忽略
        output_AGCN = self.AGCN_part(inputs_per_layer)
        output_middle = self.dropout(output_AGCN)
        output_AGCN_TCN = self.TCN_part(output_middle)

        # inputs_reshape = self.TCN_for_add_norm(inputs_per_layer)  # 暂时不判断输入输出维度是否一致，直接进行add_norm
        # output_AGCN_TCN = inputs_reshape + output_AGCN_TCN

        if (inputs_per_layer.shape[-1] == output_AGCN_TCN.shape[-1]) and (self.stride == 1):
            output_AGCN_TCN = tf.add(inputs_per_layer, output_AGCN_TCN)
        else:
            inputs_reshape = self.TCN_for_add_norm(inputs_per_layer)
            output_AGCN_TCN = inputs_reshape + output_AGCN_TCN

        return output_AGCN_TCN


class AGCN_TCN(layers.Layer):
    '''
        AGCN_TCN 搭建, 多个block构建，参数包含 out_channels, A, coff_embedding=4, kernel_size=9, stride=1
        初始化参数：
            matrix：邻接图, 需要为 tensor 类型
        输入：
            inputs_block: shape = [None, T, N, C_in]
        输出：
            output_AGCN_TCN：shape = [none, T, N, C_out]
    '''

    def __init__(self, matrix, matrix_2=None):
        super(AGCN_TCN, self).__init__()

        # 此处考虑 conv_lstm 的 attention 机制，有
        self.block_1 = AGCN_TCN_block(25, matrix)  # 变化了filters的数量（由12变为6
        # self.block_2 = AGCN_TCN_block(24, matrix)
        # self.block_3 = AGCN_TCN_block(12, matrix)


        # self.block_2 = AGCN_TCN_block(12, matrix)


        # 判断是否使用二阶图
        self.matrix_2 = matrix_2
        if self.matrix_2 is not None:  # 判断该图是否存在

            # 此处考虑 conv_lstm 的 attention 机制，有

            self.block_1_matrix_2 = AGCN_TCN_block(25, matrix_2)  # 变化了filters的数量（由12变为6
            # self.block_2_matrix_2 = AGCN_TCN_block(24, matrix_2)
            # self.block_3_matrix_2 = AGCN_TCN_block(12, matrix_2)
            self.trans_layer = layers.Conv2D(
            filters=25,  # 变化了filters的数量（由12变为6
            kernel_size=(1, 1),  # kernel_size 第一个维度为height，第二给维度为 width
            strides=(1, 1),
            padding='same',  # 这里进行填充操作, 使得和input相同的结构
            kernel_initializer=tf.keras.initializers.glorot_uniform(),
            bias_initializer=tf.keras.initializers.Zeros(),
            activation=tf.nn.leaky_relu,
            trainable=True,
            name='test'
            )

        # self.block_5 = AGCN_TCN_block(128, A, coff_embedding=4, kernel_size=9, stride=1)
        # self.block_6 = AGCN_TCN_block(128, A, coff_embedding=4, kernel_size=9, stride=1)
        # 此处 block 结果比较克制，观察结果看看

    def call(self, inputs_block):
        out = self.block_1(inputs_block)
        # out = self.block_2(out)
        # out = self.block_3(out)

        if self.matrix_2 is not None:  # 判断该图是否存在
            out_2 = self.block_1_matrix_2(inputs_block)
            # out_2 = self.block_2_matrix_2(out_2)
            # out_2 = self.block_3_matrix_2(out_2)
            out = tf.concat([out, out_2], axis=-1)
            out = self.trans_layer(out)

        # out = self.block_3(out)
        # out = self.block_4(out)
        # out = self.block_5(out)
        # out = self.block_6(out)
        return out
