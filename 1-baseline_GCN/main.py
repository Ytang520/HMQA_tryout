from AE_structure import *
from Regression import *
from tensorflow.keras import losses, optimizers, callbacks
from data_preprocessing import *
import time

# hyper_params_for_AE
epochs_AE = 1000
lr_AE = 1e-3
model_AE_check_period = 10

# sparsity_hyper_params
omiga = 0.2  # 控制 KL_divergence
beta = 0.7  # 控制稀疏项

# hyper_params_for_regression
epochs_regression = 1000
lr_regression = 1e-3
model_regression_check_period = 10

# other hyper_params
ex_choose =['1', '2', '3', '4', '5']  # 基本选定5个动作, 保留1个动作; 建议用字符串表示具体的动作，后续用字典形式表征




# 数据位置
path = None
dataset = data_preprocessing(path)  # 数据预处理, 希望尽量处理到直接可用的地步


model_AE = GCN_AE()
optimizer_AE = optimizers.Adam(lr_AE)
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='best_model_AE/best_model.hdf5',
    save_best_only=True,
    monitor='val_loss',
    mode='auto',
    period=model_AE_check_period
)

start = time.time()
for epoch in range(epochs_AE):
    for step, (x, ) in enumerate(dataset):

        with tf.GradientTape() as tape:
            inter_output, x_rec = model_AE(x)
            # !!!
            # 此处需要加入sparsity项，对每个时间步都是同一个单元要求sparsity
            # 此处应加入正则项，只是针对的是谁？——也许 layer-wise 的时候使用？
            # !!!
            loss = tf.reduce_mean(losses.mse(x, x_rec))
            # kl_divergence 约束
            inter_kl_pred = (1+tf.reduce_mean(inter_output, axis=1))/2
            loss += beta * tf.reduce_mean(losses.kl_divergence(omiga, inter_kl_pred))

        grads = tape.gradient(loss, model_AE.trainable_variables)
        optimizer_AE.apply_gradients(zip(grads, model_AE.trainable_variables))

        # 保存模型参数

        checkpoint_callback.on_epoch_end(epoch, {'val_loss': loss})
        # 此处应该是val_loss (并且不含sparsity项)，此时作为选择指标才是好的

# 计时完成
end = time.time()
print("AE_reconstrution 训练时间为: {:.2f} 秒".format(end-start))








# Regression
tf.keras.backend.set_learning_phase(0)  # 关闭dropout层, 此处dropout层不会影响，因为 tanh 是 0 均值的
model_AE.trainable = False  # 表示模型冻结，不再在后续 regression 中进行参数格更

# 依据前ex_choose, 定义好regression网络, 以及各自需要的lr (此处简化使得各regression层lr相同)
regression_networks = {}
optimizers_regression = {}

# 计算网络训练时间
start_regression = {}
end_regression = {}
elapse_regression = {}

# 集成regression训练网络的代码以简化表达
def training_for_regression(dataset, epochs, net, optimizer):
    '''
        集成 regression 网络的 training 以简化代码
    params：
        dataset: 使用的数据集
        epochs: 需要的训练参数
        net: 使用的网络
        optimizer: 使用的优化器
    '''
    for epoch in range(epochs):
        for step, (x, y) in enumerate(dataset):
            with tf.GradientTape() as tape:
                y_pred = net(x)
                loss = tf.reduce_mean(losses.huber(y, y_pred))
            grads = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(grads, net.trainable_variables))

            # 保存模型参数
            checkpoint_callback.on_epoch_end(epoch, {'val_loss': loss})
            # 此处应该是val_loss (并且不含sparsity项)，此时作为选择指标才是好的
            # 这里没有保存参数, 或许还需要再加个名称的输入参数

for name_ex in ex_choose:
    regression_networks[name_ex] = Lstm_regression()
    optimizers_regression[name_ex] = optimizers.Adam(lr_regression)

    start_regression[name_ex] = time.time()

    # 注意这里 regression 的 dataset 没设计好，应该对不同的ex有不同的dataset
    training_for_regression(dataset, epochs_regression, regression_networks[name_ex], optimizers_regression[name_ex])

    # 计算网络训练时间
    end_regression[name_ex] = time.time()
    elapse_regression[name_ex] = end_regression[name_ex] - start_regression[name_ex]


