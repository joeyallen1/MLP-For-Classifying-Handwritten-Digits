import numpy as np
import Modules

# represents the feed-forward neural network
class Sequential:

    # initialized with list of modules and loss module
    def __init__(self, modules, loss):
        self.modules = modules
        self.loss = loss

    def sgd(self, X, Y, iterations=100, lrate=0.005):
        num_points = X.shape[1]
        for iter in range(iterations):
            i = np.random.randint(0, num_points)
            Xt = X[:, i:i+1]
            Yt = Y[:, i:i+1]
            Ypred = self.forward(Xt)
            loss = self.loss.forward(Ypred, Yt)
            self.backward(self.loss.backward())
            self.sgd_step(lrate)
    
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

