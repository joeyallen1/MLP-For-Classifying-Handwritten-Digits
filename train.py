from Sequential import *
from Modules import *
import pandas as pd

train_df = pd.read_csv('MNIST Data/mnist_train.csv')
x_train = train_df.iloc[:, 1:].to_numpy()
x_train = x_train.T
y_train = train_df.iloc[:, 0].to_numpy()

test_df = pd.read_csv('MNIST Data/mnist_train.csv')
x_test = test_df.iloc[:, 1:].to_numpy()
x_test = x_test.T
y_test = test_df.iloc[:, 0].to_numpy()

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


x_train = x_train / 255
x_test = x_test / 255

print(f'Training data shape: {x_train.shape}')
print(f'Testing data shape: {x_test.shape}')
print(f'Training data labels shape: {y_train.shape}')
print(f'Testing data labels shape: {y_test.shape}')


modules = [Linear(784, 28), ReLu(),
           Linear(28,10), SoftMax()]
loss = NLL()
model = Sequential(modules, loss, x_train, x_test, y_train, y_test)

model.sgd(iterations=10000, lrate=0.001, every=100)