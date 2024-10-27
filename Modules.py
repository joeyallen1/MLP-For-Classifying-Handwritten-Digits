import numpy as np

class Module:

    # only the linear modules need to compute a step
    def sgd_step(self, lrate):
        pass


# represents the weights in a layer
class Linear(Module):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.W0 = np.random.normal(0.0, 1.0, (n, 1))  #add normally distributed noise to original weights
        self.W = np.random.normal(0.0, m ** -1, (m, n)) 

    # stores the atctivation from previous layer
    # returns pre-activation
    def forward(self, A):
        self.A = A
        return self.W.T@self.A + self.W0
    
    # stores dLdW and dLdW0 for later use in weight updates
    # returns dLdA (where A is activation from previous layer)
    def backward(self, dLdZ):
        self.dLdW = self.A@dLdZ  
        self.dLdW0 = dLdZ 
        return self.W@dLdZ
    
    
# represents negative log likelihood loss (for multiclass classification)
class NLL(Module):
    
    # returns scalar loss value given the prediction and target values
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(self.Y * np.log(self.Ypred))) * -1

    # returns dLdZ (gradient of the loss with respect to preactivation)
    # note: the NLL module is always paired with SoftMax activation
    def backward(self):
        return self.Ypred - self.Y