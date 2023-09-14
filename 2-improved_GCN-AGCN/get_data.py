from Preprocess import *
import tensorflow as tf


path = 'Kimore_ex4'
train_X, test_X, train_Y, test_Y = Data_Loader(path).import_and_preprocess_dataset()

x_train = tf.reshape(train_X, (-1, 100*25*3))
x_test = tf.reshape(test_X, (-1, 100*25*3))

def convert_to_csv(tensor_data, name):
    numpy_array = tensor_data.numpy()
    df = pd.DataFrame(numpy_array)
    df.to_csv(name + '.csv', index=False)


convert_to_csv(x_train, 'train_X')
convert_to_csv(x_test, 'test_X')
convert_to_csv(train_Y, 'Train_Y')
convert_to_csv(test_Y, 'test_Y')
