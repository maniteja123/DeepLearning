import numpy as np
from math import sqrt
from math import floor

class CNN(object):

    def __init__(self, X, y):
        inputmaps = 1
        mapsize = X.shape[:-1]

        for l in range(len(self.layers)):
            if self.layers[l].type == 's':
                mapsize = mapsize / self.layers[l].scale
                np.testing.assert_true(all(floor(mapsize)==mapsize), 'Layer ' + str(l) + ' size must be integer. Actual: '+ str(mapsize));
                for j in range(inputmaps):
                    self.layers[l].b[j] = 0

            if self.layers[l].type == 'c':
                mapsize = mapsize - self.layers[l].kernelsize + 1
                fan_out = self.layers[l].outputmaps * self.layers[l].kernelsize ** 2
                for j in range(self.layers[l].outputmaps): #  #  output map
                    fan_in = inputmaps * self.layers[l].kernelsize ^ 2;
                    for i in range(inputmaps): #  #  input map
                        self.layers[l].k[i][j] = (rand(self.layers[l].kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out))
                    self.layers[l].b[j] = 0
                inputmaps = self.layers[l].outputmaps;

        # 'onum' is the number of labels. If you have 20 labels so the output of the self.ork will be 20 neurons.
        # 'fvnum' is the number of output neurons at the last layer, the layer just before the output layer.
        # 'ffb' is the biases of the output neurons.
        # 'ffW' is the weights between the last layer and the output neurons. Note that the last layer is fully connected to the output layer, that's why the size of the weights is (onum * fvnum)
        
        fvnum = prod(mapsize) * inputmaps
        onum = y.shape[0]

        self.ffb = np.zeros((onum, 1))
        self.ffW = (np.random.rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum))
        
    def cnnff(self, X):
        n = len(self.layers)
        self.layers[0].a[0] = X
        inputmaps = 1

        for l in range(1, n):  #  for each layer
            if self.layers[l].type == 'c':
                #  !!below can probably be handled by insane matrix operations
                for j in range(self.layers[l].outputmaps):   # for each output map
                    #  create temp output map
                    k = self.layers[l].kernelsize
                    z = np.zeros_like(self.layers[l - 1].a[1].shape - [ k-1, k-1, 0])
                    for i in range(inputmaps):  #  for each input map
                        #  convolve with corresponding kernel and add to temp output map
                        z = z + convn(self.layers[l - 1].a[i], self.layers[l].k[i][j], 'valid')
                    # add bias, pass through nonlinearity
                    self.layers[l].a[j] = sigm(z + self.layers[l].b[j]);
                end
                #  set number of input maps to this layers number of outputmaps
                inputmaps = self.layers[l].outputmaps
            elif self.layers[l].type == 's':
                # downsample
                for j in range(inputmaps):
                    z = convn(self.layers[l - 1].a[j], np.ones_like(self.layers[l].scale) / (self.layers[l].scale ** 2), 'valid')
                    self.layers[l].a[j] = z[1 : -1: self.layers[l].scale, 1 : -1: self.layers[l].scale, :]

        #  concatenate all end layer feature maps into vector
        self.fv = []
        for j in range(len(self.layers[n].a)):
            self.fv = np.hstack((self.fv, np.ravel(self.layers[n].a[j]))
        #  feedforward into output perceptrons
        self.o = sigm(self.ffW * self.fv + self.ffb)

    def cnnbp(self. y)
        n = len(self.layers)

        #  error
        self.e = self.o - y;
        #  loss function
        self.L = 1/2 * np.sum(self.e ** 2) / self.e.shape[1]

        # backprop deltas
        self.od = self.e * (self.o * (1 - self.o))   #  output delta
        self.fvd = np.dot(self.ffW.T, self.od)            #  feature vector delta
        if self.layers[n].type == 'c')         #  only conv layers has sigm function
            self.fvd = self.fvd * (self.fv * (1 - self.fv))
        
        #  reshape feature vector deltas into output map style
        sa = self.layers[n].a[0].shape
        fvnum = sa(0) * sa(1)
        for j in range(len(self.layers[n].a)):
            self.layers[n].d[j] = self.fvd[((j - 1) * fvnum + 1) : j * fvnum, :].reshape(sa) #reshape(sa(0), sa(1), sa(2))

        for l in range(n - 1, 0, -1) :
            if self.layers[l].type == 'c':
                for j in range(len(self.layers[l].a)):
                    self.layers[l].d[j] = self.layers[l].a[j] * (1 - self.layers[l].a[j]) 
                        * (self.layers[l + 1].d[j].reshape([self.layers[l + 1].scale, self.layers[l + 1].scale, 1]) / self.layers[l + 1].scale ** 2)
            elif self.layers[l].type == 's':
                for i in range(len(self.layers[l].a)):
                    z = np.zeros_like(self.layers[l].a[1])
                    for j in range(len(self.layers[l + 1].a)):
                         z = z + convn(self.layers[l + 1].d[j], self.rot180(self.layers[l + 1].k[i][j]), 'full')
                    self.layers[l].d[i] = z

        #  calc gradients
        for l in range(1, n):
            if self.layers[l].type == 'c':
                for j in range(len(self.layers[l].a)):
                    for i in range(len(self.layers[l - 1].a)):
                        self.layers[l].dk[i][j] = convn(flipall(self.layers[l - 1].a[i]), self.layers[l].d[j], 'valid') / self.layers[l].d[j].shape[2]
                    self.layers[l].db[j] = np.sum(self.layers[l].d[j]) / self.layers[l].d[j].shape[2]

        self.dffW = np.dot(self.od, self.fv.T) / self.od.shape[1]
        self.dffb = np.mean(self.od, 1)

    def rot180(self, X):
        X = flipdim(flipdim(X, 0), 1)
        return X

    def cnnapplygrads(self):
        for l in range(1, len(self.layers)):
            if self.layers[l].type == 'c':
                for j in range(len(self.layers[l].a)):
                    for ii in range(len(self.layers[l - 1].a)):
                        self.layers[l].k[ii][j] = self.layers[l].k[ii][j] - self.alpha * self.layers[l].dk[ii][j]
                    self.layers[l].b[j] = self.layers[l].b[j] - opts.alpha * self.layers[l].db[j]


        self.ffW = self.ffW - self.alpha * self.dffW
        self.ffb = self.ffb - self.alpha * self.dffb

    def train(self, X, y):
        m = X.shape[2]
        numbatches = m / opts.batchsize;
        if rem(numbatches, 1) ~= 0
            error('numbatches not integer');

        net.rL = []
        for i in range(self.numepochs):
            print 'epoch ' + str(i) + '/' + str(self.numepochs)
            kk = randperm(m)
            for l in range(numbatches):
                batch_x = X[:, :, kk[(l - 1) * self.batchsize + 1 : l * self.batchsize))
                batch_y = y[:,    kk[(l - 1) * self.batchsize + 1 : l * self.batchsize))

                net = self.cnnff(batch_x)
                net = self.cnnbp(batch_y)
                net = self.cnnapplygrads()
                if isempty(net.rL):
                    net.rL = net.L
                net.rL.append(0.99 * net.rL[-1] + 0.01 * net.L)

    def test(self, X, y):
        #  feedforward
        self.cnnff(X)
        h = np.argmax(self.o, 0)
        a = np.argmax(y, 0)
        bad = np.where(h != a)
        er = len(bad) / y.shape[1]
