import numpy as np
from Modules import Module
import matplotlib.pyplot as plt


def one_hot_encode(Y: int, num_classes: int):
    '''Returns a one-hot encoding of Y as a vector with the given number of classes. '''
    encoding = np.zeros((num_classes,1))
    encoding[Y, 0] = 1
    return encoding


class Sequential:
    '''Represents a simple feed-foward neural network.'''

    def __init__(self, modules: list[Module], loss: Module, X_train: np.array, X_test: np.array, 
                 Y_train: np.array, Y_test: np.array):
        '''Initializes the network with the given list of module and loss function, as well as 
        training and testing data with target values.
        
        X_train and X_test are m by n where m is dimension of data and n is number of data points.
        Y_train and Y_test are n by 1 where n is number of target values. '''

        self.modules = modules
        self.loss = loss
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test


    def sgd(self, seed = 10, iterations=100, lrate=0.01, every=50):
        '''An implementation of stochastic gradient descent.'''

        num_points = self.X_train.shape[1]
        testing_accuracies = []
        step_numbers = []
        random_generator = np.random.default_rng(seed)
        for iter in range(iterations):
            i = random_generator.integers(0, num_points)  # choose random data point
            Xt = self.X_train[:, i:i+1]      
            Yt = self.Y_train[i,0]
            Yt = one_hot_encode(Yt, 10)
            Ypred = self.forward(Xt)    # compute forward pass
            loss = self.loss.forward(Ypred, Yt)   # compute loss
            self.backward(self.loss.backward())  # error back-propagation
            self.sgd_step(lrate)    # gradient update step

            # calculate accuracy on entire testing set for every given number of iterations
            if iter % every == 0:
                testing_accuracies.append(self.eval_accuracy())
                step_numbers.append(iter)
        
        self.plot_accuracies(step_numbers, testing_accuracies)


    def plot_accuracies(self, step_numbers: list[int], accuracies: list[float]):
        '''Plots the model's accuracy on the testing set over the training period.'''

        plt.plot(step_numbers, accuracies)
        plt.xlabel("Training Step")
        plt.ylabel("Testing Accuracy")
        plt.title("Test Accuracy During Training")
        plt.show()


    # evaluate the accuracy of the current model on the testing set
    def eval_accuracy(self):
        num_testing_points = self.X_test.shape[1]
        num_correct = 0
        for i in range(num_testing_points):      # go through all data in testing set
            Xt = self.X_test[:, i:i+1]
            Yt = self.Y_test[i, 0]
            Ypred = self.forward(Xt)      
            Ypred_class = self.modules[-1].classify(Ypred)       # find classification of prediction
            if Ypred_class == Yt:
                num_correct += 1

        return num_correct / num_testing_points            # return accuracy
    
    # does a forward pass through the network and 
    # returns final prediction
    def forward(self, Xt):
        for m in self.modules:
            Xt = m.forward(Xt)
        return Xt
    
    # error back-propagation
    def backward(self, y):
        for m in self.modules[::-1]:
            y = m.backward(y)

    # does gradient descent update for each
    # linear module in network
    def sgd_step(self, lrate):
        for m in self.modules[::-1]:
            m.sgd_step(lrate)