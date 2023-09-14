# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#
# tf.debugging.set_log_device_placement(True)
#
# # Create some tensors
# a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
# c = tf.matmul(a, b)
#
# print(c)

# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# hello = tf.constant('hello,tensorflow')
# sess= tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
# print(sess.run(hello))

# import numpy
# import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()#保证sess.run()能够正常运行
# a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
# b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
# c = tf.matmul(a, b)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# print(sess.run(c))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
#
# import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
# hello = tf.constant('hello,tensorflow')
# sess= tf.compat.v1.Session()
# print(sess.run(hello))

tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)