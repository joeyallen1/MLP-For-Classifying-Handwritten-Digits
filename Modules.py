import numpy as np

# represents a module in the neural network implementation
class Module:

    # only the linear modules need to compute a step
    def sgd_step(self, lrate):
        pass


# represents the weights in a layer
class Linear(Module):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.W0 = np.zeros((n, 1))  
        self.W = np.random.normal(0.0, float(m) ** -.5, (m, n)) #add normally distributed noise to original weights

    # stores the atctivation from previous layer
    # returns pre-activation
    def forward(self, A):
        self.A = A
        return self.W.T@A + self.W0
    
    # stores dLdW and dLdW0 for later use in weight updates
    # returns dLdA (where A is activation from previous layer)
    def backward(self, dLdZ):
        self.dLdW = self.A@dLdZ.T
        self.dLdW0 = dLdZ 
        return self.W@dLdZ
    
    # gradient descent step, updates weights
    def sgd_step(self, lrate):
        self.W = self.W - lrate * self.dLdW
        self.W0 = self.W0 - lrate * self.dLdW0
    


# represents ReLu (Rectified Linear Unit) activation function
class ReLu(Module):

    # returns activation A given pre-activation Z
    def forward(self, Z):
        self.A = np.maximum(0, Z)
        return self.A
    
    # returns dLdZ, the gradient of the loss with resepct to 
    # the pre-activation
    def backward(self, dLdA):
        dAdZ = self.A != 0  # 1 when A != 0 and 0 otherwise
        return dLdA * dAdZ
    


# represents a SoftMax activation function
class SoftMax(Module):

    # returns the activation for softmax
    def forward(self, Z):
        sum = np.sum(np.exp(Z), axis=0)
        return np.exp(Z) / sum   #issue: sum is getting way too large in later iterations
    
    # returns dLdZ (just returns input because
    # dLdZ will be calculated directly in loss module)
    def backward(self, dLdZ):
        return dLdZ
    
    # returns the final classification prediction
    # given the prediction vector
    def classify(self, Ypred):
        return np.argmax(Ypred, axis=0) 




# represents negative log likelihood loss (for multiclass classification)
class NLL(Module):
    
    # returns scalar loss value given the prediction and target values
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(-Y * np.log(Ypred)))

    # returns dLdZ (gradient of the loss with respect to preactivation)
    # note: the NLL module is always paired with SoftMax activation
    def backward(self):
        return self.Ypred - self.Y