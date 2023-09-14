import pandas as pd
from tensorflow.keras import optimizers, losses
# from AGCN_Transformer import *
from AGCN_LSTM import *
from Preprocess import *
from Graph import *
from AGCN import *
import time
# from sklearn.model_selection import train_test_split


# 参数设计
lr = 1e-4  # 目前自定义
batchsz = 10  # 目前自定义
epoch = 1000

# 数据读入和数据预处理
path = 'Kimore_ex4_v2'
train_X = tf.convert_to_tensor(pd.read_csv(path + "/train_X.csv", header=None, dtype=np.float64).values, dtype=tf.float32)
test_X = tf.convert_to_tensor(pd.read_csv(path + '/test_X.csv', header=None, dtype=np.float64).values, dtype=tf.float32)
train_Y = tf.convert_to_tensor(pd.read_csv(path + '/train_Y.csv', header=None, dtype=np.float64).values, dtype=tf.float32)
test_Y = tf.convert_to_tensor(pd.read_csv(path + '/test_Y.csv', header=None, dtype=np.float64).values, dtype=tf.float32)

train_X = tf.reshape(train_X, (-1, 100, 25, 3))
test_X = tf.reshape(test_X, (-1, 100, 25, 3))

# , test_X, train_Y, test_Y = Data_Loader(path).import_and_preprocess_dataset()
# train_Y = tf.squeeze(train_Y)  # 压缩数据，方便后续取 batch 和处理
# train_X.shape,train_Y.shape: (569, 100, 25, 3) (569,)


# 数据分 batch 处理
train_db = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_db = train_db.shuffle(42).batch(batch_size=batchsz, drop_remainder=True)   # 数据集打散 +  batch 化

# 得到 train_dataset 和 val_dataset 大小
train_size = int(0.8*int(train_X.shape[0]/batchsz))
test_size = int(train_X.shape[0]/batchsz) - train_size
train_dataset = train_db.take(train_size)
val_dataset = train_db.skip(train_size).take(test_size)
# 得到训练集和测试集


# 建立 model
AD, AD_2 = Graph(25).normalize_adjacency()
model = AGCN_LSTM(matrix=AD, matrix_2=AD_2)


# def custom_loss(x, x_rec):
#     return tf.reduce_mean(losses.Huber(delta=0.1)(x, x_rec))
#
# model.compile(loss=custom_loss, optimizer= optimizers.Adam(learning_rate=lr))

model.build(input_shape=(None, 100, 25, 3))
model.summary()
optimizer = optimizers.Adam(lr)

start_time = time.time()  # 开始时间

for epochs in range(epoch):
    loss_epoch_val = []
    loss_epoch_val_mape = []
    loss_epoch = []
    loss_epoch_mae = []
    for step, (x, y) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            y_pred = model(x)
            loss = losses.Huber(delta=0.1)(y_pred, y)

        loss_mae = tf.reduce_mean(losses.mae(y_pred, y))  # 作测试用
        loss_epoch.append(float(loss))
        loss_epoch_mae.append(loss_mae)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # evaluation
    print('epoch', epochs, 'regression_loss_huber_mean', sum(loss_epoch)/train_size)
    print('epoch', epochs, 'regression_loss_mae_mean', sum(loss_epoch_mae)/train_size)

    for step, (x, y) in enumerate(val_dataset):
        val_y = model(x)
        loss_epoch_val.append(tf.reduce_mean(losses.mae(y, val_y)))
        loss_epoch_val_mape.append(tf.reduce_mean(losses.mape(y, val_y)))

    print('epoch', epochs, 'mae_for_val', sum(loss_epoch_val)/test_size)
    print('epoch', epochs, 'mape_for_val', sum(loss_epoch_val_mape)/test_size)

    y_pred_test = model(test_X)
    print('epoch', epochs, 'mae_for_test', tf.reduce_mean(losses.mae(y_pred_test,test_Y)))
    print('epoch', epochs, 'mape_for_test', tf.reduce_mean(losses.mape(y_pred_test, test_Y)))



# 时间计时完成
end_time = time.time()
elapsed_time = end_time - start_time
print("reconstrution 训练时间为: {:.2f} 秒".format(elapsed_time))

# 测试集部分
y_pred_test = model(test_X)
print('------')
print('mae_for_test', tf.reduce_mean(losses.mae(test_Y, y_pred_test)))
print('mape_for_test', tf.reduce_mean(losses.mape(test_Y, y_pred_test)))
print('lamda')