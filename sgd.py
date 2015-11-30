import numpy as np

class BasicStochasticGradientMethod(object):
    '''
        Basic stochastic gradient descent class

        Attributes:
            w0 (np.ndarray): Starting w_0 (defaults to 0)
            eta_update (fn[int -> float]) : (optional) The eta update rule (defaults to constant 0.01)
            num_iter (int): (optional) Number of iterations of sgd to run (defaults to 100)
        '''

    def __init__(self, w0, eta_update=None, num_iter=100):
        if (eta_update == None):
            self.eta_update = lambda x: 0.01
        else:
            self.eta_update = eta_update
        self.w = w0
        self.num_iter = num_iter
        self._train_iter = 0

    def train(self, gradients):
        '''
            Train with list of gradient

            Arguments:
                self (BasicStochasticGradientMethod): Object
                gradients (List[fn[np.ndarray -> float]]): List of gradient functions at various examples
        '''
        w = self.w
        for i,g in enumerate(np.random.permutation(gradients)):
           prev_w = w
           w = w - self.eta_update(self._train_iter)*g(w)
           self._train_iter += 1
        self.w = w

    def train_one(self, gradient):
        '''
            Update with one gradient

            Arguments:
                self (BasicStochasticGradientMethod): Object
                X (np.ndarray): Matrix where every row is data point
        '''

        self.w = self.w - self.eta_update(self._train_iter)*gradient(self.w)
        self._train_iter += 1


