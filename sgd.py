import numpy as np
from numba import jit
from numpy import arange
import scipy

class BasicStochasticGradientMethod(object):
    '''
        Basic stochastic gradient descent class

        Attributes:
            w0 (np.ndarray): Starting w_0 (defaults to 0)
            eta_update (fn[int -> float]) : (optional) The eta update rule (defaults to constant 0.01)
            state_check (fn[BasicStochasticGradientMethod -> None]) : Logs current state
        '''

    def __init__(self, w0, eta_update=None, state_check=None):
        if (eta_update == None):
            self.eta_update = lambda x: 0.01
        else:
            self.eta_update = eta_update
        self.w = w0
        self._train_iter = 0
        self.state_check = state_check

    def train(self, gradients):
        '''
            Train with list of gradient

            Arguments:
                self (BasicStochasticGradientMethod): Object
                gradients (List[fn[np.ndarray -> float]]): List of gradient functions at various examples
        '''
        for g in gradients:
            prev_w = self.w
            self.train_one(g)
            if (self.state_check):
                self.state_check(self)
            if np.allclose(prev_w, self.w):
                break

    def train_one(self, gradient):
        '''
            Update with one gradient

            Arguments:
                self (BasicStochasticGradientMethod): Object
                X (np.ndarray): Matrix where every row is data point
        '''

        self.w = self.w - self.eta_update(self._train_iter)*gradient(self.w)
        self._train_iter += 1

class ShamirStochasticGradientMethod(BasicStochasticGradientMethod):
    '''
        Stochastic gradient descent class for Shamir's SGD algorithm for PCA

        Attributes:
            w0 (np.ndarray): Starting w_0 (defaults to 0)
            eta_update (fn[int -> float]) : (optional) The eta update rule (defaults to constant 0.01)
        '''

    def train_one(self, gradient):
        super(ShamirStochasticGradientMethod, self).train_one(gradient)
        self.w = self.w/np.linalg.norm(self.w)

class AlectonStochasticGradientMethod(BasicStochasticGradientMethod):
    '''
        Stochastic gradient descent class for Shamir's SGD algorithm for PCA

        Attributes:
            w0 (np.ndarray): Starting w_0 (defaults to 0)
            eta_update (fn[int -> float]) : (optional) The eta update rule (defaults to constant 0.01)
        '''

    def train_one(self, gradient):
        super(AlectonStochasticGradientMethod, self).train_one(gradient)
        normalization = scipy.linalg.sqrtm(np.linalg.pinv(self.w.T.dot(self.w)))
        self.w = self.w.dot(normalization)

