from tensorflow.keras import optimizers, losses, Sequential,regularizers
from AGCN_Transformer import *
from Preprocess import *
from Graph import *
from AGCN import *
import time
from sklearn.model_selection import train_test_split

# import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint
# from sklearn.model_selection import train_test_split

# 参数设计
lr = 1e-4  # 目前自定义
batchsz = 10  # 目前自定义
epoch = 1000

# 数据读入和数据预处理
path = 'Kimore_ex4'
train_X, test_X, train_Y, test_Y = Data_Loader(path).import_and_preprocess_dataset()
# train_Y = tf.squeeze(train_Y)  # 压缩数据，方便后续取 batch 和处理
# train_X.shape,train_Y.shape: (569, 100, 25, 3) (569,)


# 数据分 batch 处理
train_db = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
train_db = train_db.shuffle(5000).batch(batch_size=batchsz, drop_remainder=True)   # 数据集打散 +  batch 化

# 得到 train_dataset 和 val_dataset 大小
train_size = int(0.8*int(train_X.shape[0]/batchsz))
test_size = int(train_X.shape[0]/batchsz) - train_size
train_dataset = train_db.take(train_size)
val_dataset = train_db.skip(train_size).take(test_size)
# 得到训练集和测试集


# train_dataset = train_dataset.repeat(batchsz)
# train_x = iter(train_dataset)  # 此时完成一个循环，方便后续取样
# test_db = tf.data.Dataset.from_tensor_slices(X_test)


# 建立 model
type = 'Regression'
AD, AD_2 = Graph(25).normalize_adjacency()
model = AGCN_Transformer(matrix=AD, matrix_2= AD_2, type=type)

# def custom_loss(x, x_rec):
#     return tf.reduce_mean(losses.Huber(delta=0.1)(x, x_rec))
#
# model.compile(loss=custom_loss, optimizer= optimizers.Adam(learning_rate=lr))
model.build(input_shape=(None, 100, 25, 3))
model.summary()
optimizer = optimizers.Adam(lr)

start_time = time.time()  # 开始时间
# checkpoint = ModelCheckpoint("best model ex4/best_model.hdf5", monitor='val_loss', save_best_only=True,
#                               mode='auto', save_freq=1)
# history = model.fit(train_X, train_Y, validation_split=0.2, epochs=epoch,
#                          batch_size=batchsz, callbacks=[checkpoint])


if type == 'AE':
    for epoch in range(1000):
        loss_epoch = []
        loss_epoch_val = []
        for step, (x, _) in enumerate(train_dataset):

            with tf.GradientTape() as tape:
                x_rec,_ = model(x)

                # 自己目标: 对每一个时刻每一个节点的值做mse，然后取全局平均,得到reconstruction_loss
                loss = tf.reduce_mean(losses.Huber(delta=0.1)(x, x_rec))

            grads = tape.gradient(loss, model.trainable_variables)
            # tf.debugging.assert_all_finite(grads, 'Gradients contain NaN or Inf values')
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            loss_epoch.append(float(loss))

            # evaluation
        print('epoch', epoch, 'loss', sum(loss_epoch))

        for step, (x, _) in enumerate(val_dataset):
            val_x,_ = model(x)
            loss_epoch_val.append(float(tf.reduce_mean(losses.Huber(delta=0.1)(x, val_x))))

        print('epoch', epoch, 'val_loss', (loss_epoch_val))

        # test_x_rec,_ = model(test_db)
        # test_loss = tf.reduce_mean(losses.mse(test_db, test_x_rec))
        # print('test_loss', test_loss)
else:
    for epoch in range(1000):
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
        print('epoch', epoch, 'regression_loss_huber_mean', sum(loss_epoch)/train_size)
        print('epoch', epoch, 'regression_loss_mae_mean', sum(loss_epoch_mae)/train_size)

        for step, (x, y) in enumerate(val_dataset):
            val_y = model(x)
            loss_epoch_val.append(tf.reduce_mean(losses.mae(y, val_y)))
            loss_epoch_val_mape.append(tf.reduce_mean(losses.mape(y, val_y)))

        print('epoch', epoch, 'mae_for_val', sum(loss_epoch_val)/test_size)
        print('epoch', epoch, 'mape_for_val', sum(loss_epoch_val_mape)/test_size)

        y_pred_test = model(test_X)
        print('epoch', epoch, 'mae_for_test', tf.reduce_mean(losses.mae(y_pred_test,test_Y)))
        print('epoch', epoch, 'mape_for_test', tf.reduce_mean(losses.mape(y_pred_test, test_Y)))



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










#
#
# # 时间计时完成
# end_time = time.time()
# elapsed_time = end_time - start_time
# print("reconstrution 训练时间为: {:.2f} 秒".format(elapsed_time))
#
# # 得到 encoder 结果, shape=[569, 100, 8]
# fin_out, encoder_out = model.predict(train_X)
# reconstruction_loss = tf.reduce_mean(losses.mse(fin_out, train_X))
# print('数据集中reconstruction_loss平均值', tf.reduce_mean(reconstruction_loss))





# # regression 部分
# start_time = time.time()  # 开始时间
#
# # encoder_out = tf.transpose(encoder_out, [0, 2, 1])  # encoder_out.shape=[569, 100, 30]  # 10个epoch时，loss为0.8352812044482277
# regression_model = Sequential()
#
# # shape=[None, 100, 30]-> shape=[None, 10, 60] -> shape=[569, 1, 120]

#
# regression_model.compile(loss=tf.keras.losses.Huber(delta=0.1), steps_per_execution=50, optimizer= tf.keras.optimizers.Adam(learning_rate=lr))
#
# regression_model.build(input_shape=(None, 100, 30))
# regression_model.summary()
# regression_model.fit(encoder_out, train_Y, epochs=500, batch_size=batchsz, validation_split=0.2, shuffle=True)
#
# # 时间计时完成
# end_time = time.time()
# elapsed_time = end_time - start_time
# print("regression 训练时间为: {:.2f} 秒".format(elapsed_time))
#
# y_pred = regression_model.predict(encoder_out)  # 这里有过拟合的嫌疑，只是先这样做吧
# regression_loss = losses.mae(y_pred, train_Y)
# print('数据集中regression_loss: ', regression_loss, 'loss平均值', tf.reduce_mean(regression_loss))