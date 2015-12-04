from numpy.linalg import svd
from loss import EmpiricalLossFn
from sgd import *
import numpy as np
from numba import jit
from numpy import arange

class ExactPCA(object):
    '''
        Do exact PCA
    '''

    def __init__(self, numComponents, data):
        U, s, V = svd(data)
        self.pca_mat = V[:numComponents, :]

    def project(self, X):
        return self.pca_mat.T.dot(X)

class ShamirSGDPCA(object):
    '''
        Do SGD PCA
    '''

    def __init__(self, numComponents, data, step_fn = lambda x: 1e-3/(x+1)):
        w_0 = np.random.rand(*data[0].shape)
        w_0 = w_0/np.linalg.norm(w_0)
        self.step_fn = step_fn
        sub = 0*w_0
        pc = []
        losses = []
        for i in range(numComponents):
            def pca_grad(w, x):
                x = x[np.newaxis].T - sub.T
                return -x.dot(x.T).dot(w)

            def pca_objective(w, x):
                x = x[np.newaxis].T - sub.T
                return -w.dot(x.dot(x.T)).dot(w)

            w_star = np.linalg.svd(data)[-1][0]
            self.pca_grad = pca_grad
            self.pca_objective = pca_objective
            pca_loss = EmpiricalLossFn(pca_objective, pca_grad)
            objective = pca_loss.partial_eval_objective_full(data)
            target_obj = objective(w_star)
            loss_per_iter = []
            def iteration_logger(sgd):
                current_objective = objective(sgd.w)
                loss_per_iter.append(objective(sgd.w))

            sgd = ShamirStochasticGradientMethod(w_0, self.step_fn)
            gradients = [(pca_loss.partial_eval_gradient_streaming(x)) for x in data]
            sgd.train(gradients)
            pc.append(sgd.w)
            losses.append(loss_per_iter)
        self.pca_mat = np.array(pc)
        self.losses = losses

        print self.pca_mat.shape

class AlectonPCA(object):

    def __init__(self, numComponents, data, num_iter=100, step_fn = lambda x : 9e-4/(x+1)):
        Y = np.random.randn(data.shape[1], numComponents)
        def pca_grad(w, x):
                x = x[np.newaxis].T
                return -x.dot(x.T).dot(w)

        def pca_objective(w, x):
                x = x[np.newaxis].T
                return -w.dot(x.dot(x.T)).dot(w)

        self.pca_grad = pca_grad
        self.pca_objective = pca_objective
        pca_loss = EmpiricalLossFn(pca_objective, pca_grad)
        objective = pca_loss.partial_eval_objective_full(data)
        sgd = AlectonStochasticGradientMethod(Y, step_fn)
        gradients = [(pca_loss.partial_eval_gradient_streaming(x)) for x in data]
        sgd.train(gradients)
        sgd.train(gradients)
        sgd.train(gradients)
        self.pca_mat = sgd.w.T

