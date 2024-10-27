import numpy as np
import copy
import Modules

# represents the feed-forward neural network
class Sequential:

    # initialized with list of modules and loss module,
    # training and testing data, training and testing target values
    # X_train is m by n where m is dimension of data and n is number of data points
    # Y_train is 1 by n where n is number of target values
    def __init__(self, modules, loss, X_train, X_test, Y_train, Y_test):
        self.modules = modules
        self.loss = loss
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test

    # stochastic gradient descent
    def sgd(self, iterations=500, lrate=0.005, every=50):
        num_points = self.X_train.shape[1]
        testing_accuracies = []
        for iter in range(iterations):
            i = np.random.randint(0, num_points) #choose random point and target value
            Xt = self.X_train[:, i:i+1]      
            Yt = self.Y_train[:, i:i+1]
            Ypred = self.forward(Xt)    #compute forward pass
            loss = self.loss.forward(Ypred, Yt)   # compute loss 
            self.backward(self.loss.backward())  # error back-propagation
            self.sgd_step(lrate)    # gradient update step

            # calculate accuracy on testing set for every given number of iterations
            if iter % every == 0:
                testing_accuracies.append(self.eval_accuracy())
        
        # self.plot_accuracies()
        print(testing_accuracies)

    # evaluate the accuracy of the current model on the testing set
    def eval_accuracy(self):
        num_testing_points = self.X_test.shape[1]
        num_correct = 0
        for i in range(num_testing_points):      # go through all data in testing set
            Xt = self.X_test[:, i:i+1]
            Yt = self.Y_test[:, i:i+1]
            Ypred = self.forward(Xt)      
            Ypred_class = self.loss.classify(Ypred)       # find classification of prediction
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
