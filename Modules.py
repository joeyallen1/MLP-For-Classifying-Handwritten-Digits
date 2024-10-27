import numpy as np

class Module:
    def sgd_step(self, lrate):
        pass


class Linear(Module):
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.W0 = np.random.normal(0.0, 1.0, (n, 1))
        self.W = np.random.normal(0.0, m ** -1, (m, n))

    def forward(self, A):
        self.A = A
        return self.W.T@self.A + self.W0
    
    def backward(self, dLdZ):
        self.dLdW = self.A@dLdZ  
        self.dLdW0 = dLdZ 
        return self.W@dLdZ
    

class NLL(Module):
    def forward(self, Ypred, Y):
        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(self.Y * np.log(self.Ypred))) * -1

    def backward(self):
        return self.Ypred - self.Y