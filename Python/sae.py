import numpy as np
from nn import NN

class SAE(object):

    def __init__(self, sizes):
        self.sizes = sizes
    
    def setup(self ,X):
        self.ae = [None for _ in range(len(self.sizes)-1)]
        for i in range(len(self.sizes)):
            nn = NN()
            nn.setup([self.sizes[i-1], self.sizes[i], self.sizes[i]])
            self.ae[i] = nn

    def train(self, X):
        for i in range(len(self.ae)):
            self.ae[i].train(X, X)
            self.ae[i].feedforward(X, X)
            x = self.ae[i].a[1]
            # remove bias term
            X = X[:, 1:]

