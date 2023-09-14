import tensorflow as tf
from tensorflow.keras import losses


x = tf.constant([[1., 1.], [2., 2.]])
print((1+tf.reduce_mean(x, axis=0))/2)

y_true = abs(tf.random.normal((2,2), 0, 0.5, seed=5))
y_pred = tf.random.normal((2,2), 0, 0.5, seed=10)

print(losses.mse(y_true, y_pred))
# 依据走入的一个矩阵括号(二维即为行数)计算KL_divergence

# print(y_true, '----', y_pred)