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
        sum = np.sum(np.exp(Z))
        return np.exp(Z) / sum
    
    # returns dLdZ (just returns input because
    # dLdZ will be calculated directly in loss module)
    def backward(self, dLdZ):
        return dLdZ
    
    # returns the final classification prediction
    # given the prediction vector
    def classify(self, Ypred):
        return np.argmax(Ypred)




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