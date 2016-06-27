import numpy as np
import scipy as sp

class DBN(object):

    def __init__(self, sizes, numepochs, batchsize, momentum, alpha):
        self.sizes = sizes # np.array([100]) - train a 100 hidden unit RBM and visualize its weights
        self.numepochs = numepochs #1
        self.batchsize = batchsize #0
        self.momentum = momentum # 0
        self.alpha = alpha #1

    def setup(self, x):
        n = x.shape[1]
        self.sizes = np.append(n, self.sizes)
        self.estimator_ = RBM(self.alpha, self.momentum)
        self.rbm = [clone(self.estimator_) for _ in range(self.sizes)]

        for layer in range(len(dbn.sizes)):
            self.rbm.setup(self.sizes[layer], self.sizes[layer+1])

    def train(self, X):
        num = len(self.rbm)
        self.rbm[0].train(X)
        for i in range(1,num):
            X = self.rbm[i - 1].rbmup(X)
            self.rbm[i].train(X)

    def dbnunfoldtonn(self, outputsize=None):
        # DBNUNFOLDTONN Unfolds a DBN to a NN
        # dbnunfoldtonn(dbn, outputsize ) returns the unfolded dbn with a final
        # layer of size outputsize added.
        if outputsize:
            size = np.append(self.sizes, outputsize)
        else:
            size = self.sizes
        num = len(self.rbm)
        nn = NN()
        nn.setup(size)
        for i in range(num):
            nn.W[i] = np.column_stack((self.rbm[i].c, self.rbm[i].W))
