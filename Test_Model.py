from Sequential import *
from Modules import *

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1, 28 * 28).T
x_test = x_test.reshape(-1, 28 * 28).T
y_train = y_train
y_test = y_test

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


modules = [Linear(784, 5), ReLu(), SoftMax()]
loss = NLL()
model = Sequential(modules, loss, x_train, x_test, y_train, y_test)

model.sgd()