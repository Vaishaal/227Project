from sgd import BasicStochasticGradientMethod, ShamirStochasticGradientMethod
from loss import EmpiricalLossFn
from sklearn.decomposition import PCA
import numpy as np
import sklearn.datasets


def basic_sgd_test():
    objective = lambda x: x*x
    gradient = lambda x: x*2
    sgd = BasicStochasticGradientMethod(34)
    gradients = [gradient for i in range(10000)]
    sgd.train(gradients)
    assert(np.isclose(sgd.w, 0, rtol=1e-6, atol=1e-6))

    sgd2 = BasicStochasticGradientMethod(34, lambda x: 0.1)
    gradients = [gradient for i in range(10000)]
    sgd2.train(gradients)
    assert(np.isclose(sgd2.w, 0, rtol=1e-6, atol=1e-6))
    assert(sgd2._train_iter <= sgd._train_iter)


def sgd_changing_stepsize_test():
    objective = lambda x: x*x
    gradient = lambda x: x*2
    sgd = BasicStochasticGradientMethod(34, lambda x: 1.0/(x+1) )
    gradients = [gradient for i in range(10)]
    sgd.train(gradients)
    assert(np.isclose(sgd.w, 0, rtol=1e-6, atol=1e-6))

def hinge_loss_test():
    def objective(w, x):
        y = x[-1]
        x_ = x[:-1]
        return max(0, 1 - x_.dot(w)*y)

    def gradient_fn(w, x):
        y = x[-1]
        x_ = x[:-1]
        if (y*w.dot(x_) < 1):
            return -x_*y
        else:
            return np.zeros(x_.shape)

    hinge_loss = EmpiricalLossFn(objective, gradient_fn)
    digits = sklearn.datasets.load_digits(2)
    data = digits.data
    labels = (digits.target*2) - 1
    X = np.hstack((data,labels[np.newaxis].T))
    objective = hinge_loss.partial_eval_objective_full(X)
    gradients = [hinge_loss.partial_eval_gradient_streaming(x) for x in X]
    sgd = BasicStochasticGradientMethod(np.zeros(data[0].shape), lambda x: 1)

    # Do 3 passes JIC
    sgd.train(gradients)
    sgd.train(gradients)
    sgd.train(gradients)

    assert objective(sgd.w) == 0
    assert sum(np.sign(data.dot(sgd.w)) == labels) == 360

def pca_loss_test():
    #This is nonconvex!!!

    digits = sklearn.datasets.load_digits(2)
    X = digits.data
    w_star = np.linalg.svd(X)[-1][0]

    def pca_grad(w, x):
        x = x[np.newaxis].T
        return -x.dot(x.T).dot(w)

    def pca_objective(w, x):
        x = x[np.newaxis].T
        return -w.dot(x.dot(x.T)).dot(w)

    pca_loss = EmpiricalLossFn(pca_objective, pca_grad)
    objective = pca_loss.partial_eval_objective_full(X)
    gradients = [pca_loss.partial_eval_gradient_streaming(x) for x in X]
    shuffled_gradients = np.random.permutation(gradients)

    w_0 = np.random.rand(*X[0].shape)
    w_0 = w_0/np.linalg.norm(w_0)

    sgd = ShamirStochasticGradientMethod(w_0, lambda x: 1e-3/(x+1))
    sgd.train(np.random.permutation(gradients), False)
    sgd.train(np.random.permutation(gradients), False)
    assert np.linalg.norm(sgd.w - w_star) < 0.01
