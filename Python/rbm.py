import numpy as np
import scipy as sp
from sklearn.utils import gen_even_slices
from utils import sigm
from utils import sigmrnd

class RBM(object):

    def __init__(self, alpha, momentum):
        self.alpha = alpha
        self.momentum = momentum
        
    def setup(self, n_visible, n_hidden):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.W = np.zeros(n_hidden, n_visible) #n_hidden X n_visible
        self.vW = np.zeros(n_hidden, n_visible) #n_hidden X n_visible
        self.b = np.zeros(n_visible, 1) # n_visible X 1
        self.vb = np.zeros(n_visible, 1) # n_visible X 1
        self.c = np.zeros(n_hidden, 1) #n_hidden X 1
        self.vc = np.zeros(n_hidden, 1) #n_hidden X 1
        self.batchsize = None

    def rbmdown(self, X):
        X = sigm(np.dot(X, self.W) + self.b)
        return X

    def rbmup(self, X):
        X = sigm(np.dot(X, self.W.T) + self.c)
        return X

    def train(self, X):
        n_samples, n_features = X.shape
        
        for i in range(self.numepochs):
            kk = np.random.permutation(m)
            err = 0

            for l in gen_even_slices(n_samples, self.batchsize):
                batch = x[l, :]
                v1 = batch # n_samples X n_visible
                h1 = sigmrnd(np.dot(v1, self.W.T) + self.c)  # n_samples X n_hidden
                v2 = sigmrnd(np.dot(h1, self.W) + self.b) # n_samples X n_visible
                h2 = sigm(np.dot(v2, self.W.T) + self.c) # n_samples X n_hidden

                c1 = np.dot(h1.T, v1) # n_hidden X n_visible
                c2 = np.dot(h2.T, v2) # n_hidden X n_visible

                self.vW = self.momentum * self.vW + self.alpha * (c1 - c2) / self.batchsize  # n_hidden X n_visible
                self.vb = self.momentum * self.vb + self.alpha * np.sum(v1 - v2, axis=0) / self.batchsize # n_visible X 1
                self.vc = self.momentum * self.vc + self.alpha * np.sum(h1 - h2, axis=0) / self.batchsize # n_hidden X 1

                self.W = self.W + self.vW # n_hidden X n_visible
                self.b = self.b + self.vb # n_visible X 1
                self.c = self.c + self.vc # n_visible X 1

                err = err + np.sum(np.power(v1 - v2), 2) / self.batchsize
            
            print 'epoch '+ str(i) + '/' + str(self.numepochs) + '. Average reconstruction error is: ' + str(err / numbatches)
        
