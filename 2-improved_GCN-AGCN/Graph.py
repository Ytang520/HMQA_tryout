import tensorflow as tf
import numpy as np


# from IPython.core.debugger import set_trace

class Graph():
    '''
    得到邻接图，暂不考虑 bias_graph 偏置图
    参数：
        num_node: 表示所有的节点数量，指定为25个
    输入：无
    输出：
        A 一阶邻接图
        A2 二阶邻接图
    '''

    def __init__(self, num_node=25):
        self.num_node = num_node
        self.AD, self.AD2 = self.normalize_adjacency()

    def normalize_adjacency(self):
        self_link = [(i, i) for i in range(self.num_node)]
        neighbor_1base = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21),
                          (6, 5), (7, 6), (8, 7), (9, 21), (10, 9),
                          (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
                          (16, 15), (17, 1), (18, 17), (19, 18), (20, 19),
                          (22, 23), (23, 8), (24, 25), (25, 12)]
        neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        edge = self_link + neighbor_link
        A = np.zeros((self.num_node, self.num_node), dtype=np.float32)  # 一阶 adjacency matrix，并且自己对自己也算一阶矩阵
        for i, j in edge:
            A[j, i] = 1
            A[i, j] = 1

        A2 = np.zeros((self.num_node, self.num_node), dtype=np.float32)  # second order adjacency matrix

        for root in range(A.shape[1]):  # 目前暂时如此处理二阶邻接阵，后续可采用邻接阵乘法+一些简单处理得到
            for neighbour in range(A.shape[0]):
                if A[root, neighbour] == 1:
                    for neighbour_of_neigbour in range(A.shape[0]):
                        if A[neighbour, neighbour_of_neigbour] == 1:
                            A2[root, neighbour_of_neigbour] = 1

        # A2 表示二阶邻接图，可能包含 类一阶关系的二阶关系

        # 归一化处理
        A = self.normalize_digraph(A)
        A2 = self.normalize_digraph(A2)

        # A = tf.nn.softmax(A, axis=-1)  # 对每一行处理softmax, 得到零阶矩阵；原始代码采用的也是行归一化
        # A2 = tf.nn.softmax(A2, axis=-1)  # 对每一行处理softmax, 得到零阶矩阵；原始代码采用的也是行归一化

        # 转化为 tensor 形式
        AD = tf.convert_to_tensor(A, dtype=tf.float32)
        AD2 = tf.convert_to_tensor(A2, dtype=tf.float32)

        return AD, AD2

    def normalize_digraph(self, A):  # 除以每行的和
        Dl = np.sum(A, 1)
        h, w = A.shape
        Dn = np.zeros((w, w))
        for i in range(w):
            for j in range(w):
                if Dl[i] > 0:
                    Dn[i, j] = A[i, j] * Dl[i] ** (-1)
        return Dn
