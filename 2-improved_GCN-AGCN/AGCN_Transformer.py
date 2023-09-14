import tensorflow as tf
from AGCN import *
from Transformer import *


class AGCN_Transformer(tf.keras.Model):
    '''
    AGCN和Transformer模块拼接
    参数：
        matrix：原始邻接图
        matrix_2=None: 二阶邻接图, 为可选项
        type='AE': 表示是否使用AE结构，可选输入为 'AE' 和 ‘Regression’
    输入：
        inputs: shape = [None, T, N, C_in]
    输出：
        output_AGCN_TCN：shape = [None, T, N, C_in]
    '''

    def __init__(self, matrix, matrix_2=None, type='AE'):
        super(AGCN_Transformer, self).__init__()
        self.type = type
        self.AGCN = AGCN_TCN(matrix=matrix, matrix_2=matrix_2)
        self.Transformer = Transformer(type=self.type)
        if self.type == 'AE':
            self.type_choose = True
        else:
            self.type_choose = False

    def call(self, inputs):
        out = self.AGCN(inputs)
        if self.type_choose:
            out, encoder_out = self.Transformer(out)
            return out, encoder_out
        else:
            out = self.Transformer(out)
            return out

        # 直接得到regression结果的代码



