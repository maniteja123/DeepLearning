import numpy as np
from utils import sigm, tanh, linear, softmax
from math import sqrt

act = {'sigm' : sigm, 'tanh' : tanh}
out = {'sigm': sigm, 'softmax': softmax, 'linear': linear}

class NN(object):

    def __init__(self, sizes):
        self.sizes = sizes
        self.n_layers = len(sizes) 
        self.learningRate = 2  # Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
        self.momentum = 0.5
        self.scaling_learningRate = 1  # Scaling factor for the learning rate (each epoch)
        self.weightPenaltyL2 = 0 # L2 regularization
        self.nonSparsityPenalty  = 0 #  Non sparsity penalty
        self.sparsityTarget = 0.05 #  Sparsity target
        self.dropoutFraction = 0 # Dropout level
        self.activation_function = 'tanh_opt' # Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh' (optimal tanh).
        self.output = 'sigm'  # output unit 'sigm' (=logistic), 'softmax' and 'linear'
        self.testing = False    
        self.W = [None for _ in range(1, self.n_layers)]
        self.vW = [None for _ in range(1, self.n_layers)]
        self.p = [None for _ in range(1, self.n_layers)]
        self.n_outputs = self.sizes[-1]

        for i in range(1, self.n_layers):
            # weights and weight momentum
            # +1 in shape for bias
            self.W[i - 1] = (np.random(self.sizes[i], self.sizes[i - 1]+1) - 0.5) * 2 * 4 * sqrt(6 / (self.sizes[i] + self.sizes[i - 1])) 
            self.vW[i - 1] = np.zeros_like(self.W[i - 1])

            # average activations
            self.p[i]= np.zeros(1, self.sizes[i])

    def feedforward(self, X, y):
        n_samples, n_features = X.shape
        X = np.column_stack((np.ones(n_samples), X))
        self.a = [None for _ in range(self.n_layers)]
        self.a[0] = X # n_samples X  n_inputs
        self.dropOutMask = [None for _ in range(1, self.n_layers)]

        for i in range(1, self.n_layers-1):
            act_fun = act[self.activation_function]
            self.a[i] = act_fun(np.dot(self.a[i - 1], self.W[i - 1].T)) # n_samples X n_hidden

            # dropout
            if self.dropoutFraction > 0 :
                if self.testing:
                    self.a[i] = (1 - self.dropoutFraction) * self.a[i]
                else:
                    self.dropOutMask[i] = rand(self.a[i].shape) > self.dropoutFraction
                    self.a[i] = self.dropOutMask[i] * self.a[i]  

            # calculate running exponential activations for use with sparsity
            if self.nonSparsityPenalty>0:
                self.p[i] = 0.99 * self.p[i] + 0.01 * np.mean(self.a[i], axis=0) # 1 X n_hidden

            self.a[i] = np.column_stack((np.ones(n_samples), self.a[i]))

        out_fun = out[self.output]
        output_layer = self.n_layers - 1
        self.a[output_layer] = out(np.dot(self.a[output_layer - 1], self.W[output_layer - 1].T))  #n_samples X n_outputs
        self.e = y - self.a[output_layer]

    def backprop(self):
        sparsityError = 0
        self.dW = [None for _ in range(self.n_layers)]
        d = [None for _ in range(self.n_layers)]
        output_layer = self.n_layers - 1
        if self.output == 'sigm':
            d[output_layer] = - self.e[output_layer] * self.a[output_layer] * (1 - self.a[output_layer]) # n_samples X n_outputs
        elif self.output == 'linear':
            d[output_layer] = - self.e[output_layer]
        n_samples = d[output_layer].shape[0]

        for i in range(self.n_layers - 2, 0, -1):
            if self.activation_function == 'sigm':
                d_act = self.a[i] * (1 - self.a[i])  # n_samples X n_visible
            elif self.activation_function == 'tanh':
                d_act = 1.7159 * 2/3 * (1 - 1/(1.7159)**2 * self.a[i] **2)

            n_sampless = self.a[i].shape[0]
            if self.nonSparsityPenalty>0:
                pi = self.p[i]
                sparsityError = np.column_stack(np.zeros((n_samples, 1)),
                        self.nonSparsityPenalty * (-self.sparsityTarget / pi + (1 - self.sparsityTarget) / (1 - pi))) # n_samples X n_visible
            
            # Backpropagate first derivatives
            if i+1==output_layer: #  in this case in d{n} there is not the bias term to be removed             
                d[i] = ( np.dot(d[i + 1], self.W[i]) + sparsityError ) * d_act # n_samples X n_visible
            else: # in this case in d{i} the bias term has to be removed
                d[i] = ( np.dot(d[i + 1][:,1:], self.W[i]) + sparsityError) * d_act;
        
            if self.dropoutFraction>0:
                d[i] = d[i] * np.column_stack(np.ones(n_samples, 1), self.dropOutMask[i])

        for i in range(self.n_layers - 1):
            if i+1 == output_layer:
                self.dW[i] = np.dot(d[i + 1].T, self.a[i]) / n_samples # n_hidden X n_visible
            else:
                self.dW[i] = np.dot(d[i + 1][:,1:].T, self.a[i]) / n_samples


    def applyGrad(self):
        for i in range(self.n_layers - 1):
            n_samples = self.W[i].shape[0]
            if self.weightPenaltyL2 > 0:
                dW = self.dW[i] + self.weightPenaltyL2 * np.column_stack((np.zeros(n_samples,1) ,self.W[i][:, 1:]))
            else:
                dW = self.dW[i]
            
            dW = self.learningRate * dW;
            
            if self.momentum>0:
                self.vW[i] = self.momentum * self.vW[i] + dW
                dW = self.vW[i]
                
            self.W[i] = self.W[i] - dW
