from sgd import BasicStochasticGradientMethod
import numpy as np

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
    assert(sgd2._train_iter < sgd._train_iter)


def sgd_changing_stepsize_test():
    objective = lambda x: x*x
    gradient = lambda x: x*2
    sgd = BasicStochasticGradientMethod(34, lambda x: 1.0/(x+1) )
    gradients = [gradient for i in range(10)]
    sgd.train(gradients)
    assert(np.isclose(sgd.w, 0, rtol=1e-6, atol=1e-6))



