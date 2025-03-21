import numpy as np



class Module:
    '''Represents a module in the neural network implementation. Includes 
    linear modules, layer activation functions, and output activation.'''


    def sgd_step(self, lrate: float):
        '''Only linear modules have weights that need to be updated, all other modules don't do anything for this step.'''

        pass



class Linear(Module):
    '''Represents a layer of linear modules.'''

    def __init__(self, m: int, n: int, seed = 10):
        '''Initializes the weights in this layer using the given random seed. 
        n is the number of neurons in this layer and m is the number of weights 
        associated with each neuron. m should match the dimension of the input data.'''

        self.m = m
        self.n = n
        self.W0 = np.zeros((n, 1))  
        random_generator = np.random.default_rng(seed)
        self.W = random_generator.normal(0.0, float(m) ** -.5, (m, n))   # add normally distributed noise to original weights


    def forward(self, A: np.ndarray) -> np.ndarray:
        '''Applies a linear transformation to the input from previous layer.
        Stores the given input for use during back-propagation.'''

        self.A = A
        return self.W.T@A + self.W0
    

    def backward(self, dLdZ: np.ndarray) -> np.ndarray:
        '''Returns the gradient of the loss with respect to the activation from previous layer.
        Stores the gradient of the loss with respect to the weights for use during 
        gradient update steps.'''

        self.dLdW = self.A@dLdZ.T
        self.dLdW0 = dLdZ
        return self.W@dLdZ
    

    def sgd_step(self, lrate: float):
        '''Implements a gradient descent update by updating this module's weights given a learning rate.'''

        self.W = self.W - lrate * self.dLdW
        self.W0 = self.W0 - lrate * self.dLdW0
    


class ReLu(Module):
    '''Represents a ReLu (Rectified Linear Unit) activation function.'''


    def forward(self, Z: np.ndarray) -> np.ndarray:
        '''Returns the activation given pre-activation Z.'''

        self.A = np.maximum(0, Z)
        return self.A
    
   
    def backward(self, dLdA: np.ndarray) -> np.ndarray:
        '''Computes and returns the gradient of the loss with respect to the pre-activation
        (values passed from previous layer).'''

        dAdZ = self.A != 0  # 1 when A != 0 and 0 otherwise
        return dLdA * dAdZ
    


class SoftMax(Module):
    '''Represents a SoftMax activation function (interpreted as probability of each possible classification).'''


    def forward(self, Z: np.ndarray) -> np.ndarray:
        '''Returns the activation for softmax given the pre-activation.'''

        sum = np.sum(np.exp(Z), axis=0)
        return np.exp(Z) / sum   
    

    def backward(self, dLdZ):
        '''Returns the gradient of the loss with respect to the pre-activation.
        This just returns the input because the loss module directly calculates the needed value
        and passes it to this method (because SoftMax is always paired with NLL loss in this implementation.)'''

        return dLdZ
    

    def classify(self, Ypred: np.ndarray) -> int:
        '''Returns the final classification prediction given the prediction vector (which 
        should be the output of the forward method of this module).'''

        return int(np.argmax(Ypred, axis=0))



class NLL(Module):
    '''Represents a module for Negative Log Likelihood loss (used for multiclass classification).'''
    
    
    def forward(self, Ypred: np.ndarray, Y: np.ndarray) -> float:
        '''Returns a scalar loss value given the prediction and target vectors.
        Also stores the inputs for use during back-propagation.'''

        self.Ypred = Ypred
        self.Y = Y
        return float(np.sum(-Y * np.log(Ypred)))
    

    def backward(self) -> np.ndarray:
        '''Returns the gradient of the loss with respect to the preactivation (not activation, 
        since this NLL module is always paired with softmax activation.)'''

        return self.Ypred - self.Y