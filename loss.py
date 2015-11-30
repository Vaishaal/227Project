import numpy as np

class EmpiricalLossFn(object):
    '''
        Represents an separable function F(w, X) = \sum_{i=0}^{N} (f(w,X_i))

        Attributes:
           objective (fn[(np.ndarray, np.ndarray) -> float]) : Objective function
           grad_fn  (fn[(np.ndarray, np.ndarray) -> np.ndarray]): Gradient of objective
    '''

    def __init__(self, objective, grad_fn):
        self.objective = objective
        self.grad_fn = grad_fn

    def partial_eval_objective_full(self, X):
        '''
            Return a univariate function that takes in w and computes
            fn(w, X)

            Arguments:
                X (np.ndarray): Data matrix every row represents data point
        '''

        return lambda w: np.sum(np.apply_along_axis(lambda x: self.objective(w, x), axis=1, arr=X))

    def partial_eval_gradient_full(self, X):
        '''
            Return a univariate function that takes in w and computes
            \nabla_{w} fn(w, X)

            Arguments:
                X (np.ndarray): Data matrix every row represents data point
        '''
        return lambda w: np.sum(np.apply_along_axis(lambda x: self.grad_fn(w, x), axis=1, arr=X), axis=0)

    def partial_eval_gradient_streaming(self, x):
        '''
            Takes in one data point and weturn a univariate function that takes in w and computes
            \nabla_{w} fn(w, x).

            Arguments:
                x (np.ndarray): one data point
        '''
        return lambda w: self.grad_fn(w, x)

