from sgd import BasicStochasticGradientMethod, ShamirStochasticGradientMethod
from loss import EmpiricalLossFn
from sklearn.decomposition import PCA
import numpy as np
import sklearn.datasets
import argparse
import matplotlib.pyplot as plt

experiments = ["shamirPCA", "strictSadddlePCA", "strictSaddleTensorDecomp", "strictSadddleNeuralNet"]

def shamirPCA():
    digits = sklearn.datasets.load_digits()
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

    loss_per_iter = []
    power_loss_per_iter = []

    def iteration_logger(sgd):
        print "Sgd Iteration: {0}, Loss: {1}".format(sgd._train_iter, objective(sgd.w))
        loss_per_iter.append(objective(sgd.w))

    sgd = ShamirStochasticGradientMethod(w_0, lambda x: 1e-3/(x+1), iteration_logger)
    sgd.train(np.random.permutation(gradients))

    for i in range(len(loss_per_iter)):
        w_p =  X.T.dot(np.random.rand(*X[:,0].shape))
        w_p = w_p/np.linalg.norm(w_p)
        print "Power Iteration: {0}, Loss: {1}".format(i, objective(w_p))
        power_loss_per_iter.append(objective(w_p))

    print power_loss_per_iter
    plt.figure()
    plt.plot(zip(loss_per_iter, power_loss_per_iter))
    plt.show()



def strictSaddlePCA():
    pass

def strictSaddleTensorDecomp():
    pass

def strictSaddleNeuralNet():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run 227BT sgd experiments')
    parser.add_argument('experiment', type=str, help='Which experiment to run: \
            {0}'.format(experiments))
    args = parser.parse_args()
    if args.experiment in experiments:
        eval(args.experiment + "()")

