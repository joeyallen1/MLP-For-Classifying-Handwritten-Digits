import numpy as np
import Modules

# represents the feed-forward neural network
class Sequential:

    # initialized with list of modules and loss module
    def __init__(self, modules, loss):
        self.modules = modules
        self.loss = loss

    # stochastic gradient descent
    def sgd(self, X, Y, iterations=100, lrate=0.005):
        num_points = X.shape[1]
        for iter in range(iterations):
            i = np.random.randint(0, num_points) #choose random point and target value
            Xt = X[:, i:i+1]      
            Yt = Y[:, i:i+1]
            Ypred = self.forward(Xt)    #compute forward pass
            loss = self.loss.forward(Ypred, Yt)   # compute loss 
            self.backward(self.loss.backward())  # error back-propagation
            self.sgd_step(lrate)    # gradient update step
    
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

