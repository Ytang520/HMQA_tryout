import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
# from IPython.core.debugger import set_trace
from sklearn.model_selection import train_test_split
import numpy as np

index_Spine_Base = 0
index_Spine_Mid = 4
index_Neck = 8
index_Head = 12  # no orientation
index_Shoulder_Left = 16
index_Elbow_Left = 20
index_Wrist_Left = 24
index_Hand_Left = 28
index_Shoulder_Right = 32
index_Elbow_Right = 36
index_Wrist_Right = 40
index_Hand_Right = 44
index_Hip_Left = 48
index_Knee_Left = 52
index_Ankle_Left = 56
index_Foot_Left = 60  # no orientation
index_Hip_Right = 64
index_Knee_Right = 68
index_Ankle_Right = 72
index_Foot_Right = 76  # no orientation
index_Spine_Shoulder = 80
index_Tip_Left = 84  # no orientation
index_Thumb_Left = 88  # no orientation
index_Tip_Right = 92  # no orientation
index_Thumb_Right = 96  # no orientation


class Data_Loader():
    '''
    本部分完成数据导入和预处理
    参数：
        dir：数据路径
    输出：
        输出顺序：train_x, test_x, train_y, test_y
        train_x : shape=[train_num, T, Nodes, channel], train_num 表示训练动作数，为455
        test_x : shape=[test_num, T, Nodes, channel], test_num 表示测试动作数
        train_y : shape=[train_num, 1]: 表示训练动作数的评分 —— 尾部会自动生成1的维度
        test_y : shape=[test_num, 1], test_num 表示测试动作数的评分 —— 尾部会自动生成1的维度
    '''

    def __init__(self, dir):
        self.dir = dir
        self.body_part = self.body_parts()
        self.num_joints = len(self.body_part)
        self.sc = StandardScaler()
        self.sc_test = StandardScaler()
        self.sc_y = StandardScaler()
        # self.sc_test_y = StandardScaler()

    def body_parts(self):
        body_parts = [index_Spine_Base, index_Spine_Mid, index_Neck, index_Head, index_Shoulder_Left, index_Elbow_Left,
                      index_Wrist_Left, index_Hand_Left, index_Shoulder_Right, index_Elbow_Right, index_Wrist_Right,
                      index_Hand_Right, index_Hip_Left, index_Knee_Left, index_Ankle_Left, index_Foot_Left,
                      index_Hip_Right, index_Knee_Right, index_Ankle_Right, index_Foot_Right, index_Spine_Shoulder,
                      index_Tip_Left, index_Thumb_Left, index_Tip_Right, index_Thumb_Right
                      ]
        # Bodyparts 里原有2个index_ankle_right, 现已处理

        return body_parts

    def import_and_preprocess_dataset(self):
        # train_x = pd.read_csv(self.dir + "\Train_X.csv", header=None).iloc[:, :].values
        # train_y = pd.read_csv(self.dir + "\Train_Y.csv", header=None).iloc[:, :].values

        # 取数据，这里 train_x 数据比较特殊，第四列为无用数据
        train_x = pd.read_csv(self.dir + "\Train_X.csv", header=None, dtype=np.float64)  # 应该不必iloc
        train_y = pd.read_csv(self.dir + "\Train_Y.csv", header=None, dtype=np.float64)
        col_drop = [4 * i + 3 for i in range(int(train_x.shape[1] / 4))]
        train_x.drop(train_x.columns[col_drop], axis=1, inplace=True)

        # 数据预处理
        train_x = train_x.values
        train_y = train_y.values
        train_y = self.sc_y.fit_transform(train_y)

        # train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
        # train_y = tf.convert_to_tensor(train_y/50, dtype=tf.float32)  # 这里除以50.保证标签在0~1之间
        # np.reshape(np.transpose(train_x), )
        train_x = np.reshape(train_x, [-1, 100, train_x.shape[-1]])
        train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=10000000)  # 数据集分割

        # [N, T, nodes*channel] -> [T, N*nodes*channels]
        train_x = np.reshape(np.transpose(train_x, [1, 0, 2]), [100, -1])
        test_x = np.reshape(np.transpose(test_x, [1, 0, 2]), [100, -1])

        # 转化为tensor
        train_x = tf.convert_to_tensor(train_x, dtype=tf.float32)
        # train_y = tf.convert_to_tensor(train_y / 50, dtype=tf.float32)  # 这里除以50.保证标签在0~1之间
        train_y = tf.convert_to_tensor(train_y, dtype=tf.float32)  # 这里除以50.保证标签在0~1之间
        test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
        test_y = tf.convert_to_tensor(test_y, dtype=tf.float32)  # 这里除以50.保证标签在0~1之间
        # test_y = tf.convert_to_tensor(test_y/50, dtype=tf.float32)  # 这里除以50.保证标签在0~1之间


        # 正规化和处理(对 train_x 依每个人(样本，100个时间)每个节点每个channel按时间序列进行标准化处理，不对 train_y 标准化处理)
        train_x = self.sc.fit_transform(train_x)
        test_x = self.sc_test.fit_transform(test_x)

        # train_y = self.sc_y.fit_transform(train_y)
        # test_y = self.sc_test_y.fit_transform(test_y)
        # 用linear作regression，并且使用standard_norm

        # 返回到原来状态
        # [T, N*nodes*channels] -> [N, T, Nodes, Channel]
        train_x = tf.transpose(tf.reshape(train_x, [100, -1, self.num_joints, 3]), [1, 0, 2, 3])
        test_x = tf.transpose(tf.reshape(test_x, [100, -1, self.num_joints, 3]), [1, 0, 2, 3])

        return train_x, test_x, train_y, test_y
