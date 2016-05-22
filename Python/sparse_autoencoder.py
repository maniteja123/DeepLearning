import numpy as np
import scipy.optimize
import scipy.io
import matplotlib.pyplot as plt
import time
import sys
import math

class SparseAutoEncoder(object):

    def __init__(self, visible_size, hidden_size, sparsity_param, lambda_decay, beta):
    
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.lamda = lambda_decay
        self.rho_param = sparsity_param
        self.beta = beta
        
        r = math.sqrt(6) / math.sqrt(visible_size + hidden_size + 1)
        
        rand = np.random.RandomState(int(time.time()))
        
        W1 = np.asarray(rand.uniform(-r, r, (hidden_size, visible_size)))
        W2 = np.asarray(rand.uniform(-r, r, (visible_size, hidden_size)))
        
        b1 = np.zeros((hidden_size, 1))
        b2 = np.zeros((visible_size, 1))

        self.theta = np.concatenate((W1.flatten(), W2.flatten(),
                                        b1.flatten(), b2.flatten()))

    def cost(self, theta, X):
    
        n_features, n_samples = X.shape
        W1 = theta[0 : self.hidden_size * self.visible_size].reshape(self.hidden_size, self.visible_size)
        W2 = theta[self.hidden_size * self.visible_size : 2 * self.hidden_size * self.visible_size].reshape(self.visible_size, self.hidden_size)
        b1 = theta[2 * self.hidden_size * self.visible_size : 2 * self.hidden_size * self.visible_size +self.hidden_size].reshape(self.hidden_size, 1)
        b2 = theta[2 * self.hidden_size * self.visible_size + self.hidden_size :].reshape(self.visible_size, 1)
        
        hidden_layer = self.sigmoid(np.dot(W1, X) + b1)
        output_layer = self.sigmoid(np.dot(W2, hidden_layer) + b2)
        
        rho_cap = np.sum(hidden_layer, axis = 1) / n_samples
        
        errors = output_layer - X
        
        sse = 0.5 * np.sum(np.multiply(errors, errors)) / n_samples
        
        reg = 0.5 * self.lamda * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
        
        cost_sparse = sse + reg + self.beta * self.kl(self.rho_param, rho_cap)

        KL_div_grad = self.beta * self.kl_grad(self.rho_param, rho_cap)

        delta_output = np.multiply(errors, np.multiply(output_layer, 1 - output_layer))

        delta_hidden = np.multiply(np.dot(W2.T, delta_output) + np.matrix(KL_div_grad).T, np.multiply(hidden_layer, 1 - hidden_layer))

        W1_grad = np.dot(delta_hidden, X.T) / n_samples + self.lamda * W1
        W2_grad = np.dot(delta_output, hidden_layer.T) / n_samples + + self.lamda * W2
        b1_grad = np.sum(delta_hidden, axis=1) / n_samples
        b2_grad = np.sum(delta_output, axis=1) / n_samples

        W1_grad = np.array(W1_grad)
        W2_grad = np.array(W2_grad)
        b1_grad = np.array(b1_grad)
        b2_grad = np.array(b2_grad)

        theta_grad = np.concatenate((W1_grad.flatten(), W2_grad.flatten(), b1_grad.flatten(), b2_grad.flatten()))
        return (cost_sparse, theta_grad)

    def sigmoid(self, X):
        return (1 / (1 + np.exp(-X)))
        
    def kl(self, rho_param, rho_cap):
        return np.sum(rho_param * np.log(rho_param / rho_cap) + (1 - rho_param) * np.log((1 - rho_param) / (1 - rho_cap)))

    def kl_grad(self, rho_param, rho_cap):
        return (-(rho_param / rho_cap) + ((1 - rho_param) / (1 - rho_cap)))
        
def normalizedata(data):
    data = data - np.mean(data)
    std_dev = 3 * np.std(data)
    data = np.maximum(np.minimum(data, std_dev), -std_dev) / std_dev 
    data = (data + 1) * 0.4 + 0.1
    return data

def loaddata(num_patches, patch_side):

    images = scipy.io.loadmat(sys.path[0]+'/IMAGES.mat')
    images = images['IMAGES']
    
    data = np.zeros((patch_side*patch_side, num_patches))
    
    rand = np.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size = (num_patches, 2))
    image_number  = rand.randint(10, size = num_patches)


    for i in xrange(num_patches):    
        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]
       
        patch = images[index1: index1+patch_side, index2: index2+patch_side, index3]
        patch = patch.flatten()
        data[:, i] = patch
    
    data = normalizedata(data)
    return data

def visualizeW(opt_W, vis_patch_side, hid_patch_side):
   
    figure, axes = plt.subplots(nrows = hid_patch_side,  ncols = hid_patch_side)                                        

    index = 0                                         
    for axis in axes.flat:

        image = axis.imshow(opt_W[index, :].reshape(vis_patch_side, vis_patch_side),
                            cmap = plt.cm.gray, interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
        
    plt.show()

def executeSparseAutoencoder():

    """ Define the parameters of the Autoencoder """
    
    vis_patch_side = 8      # side length of sampled image patches
    hid_patch_side = 5      # side length of representative image patches
    rho            = 0.01   # desired average activation of hidden units
    lamda          = 0.0001 # weight decay parameter
    beta           = 3      # weight of sparsity penalty term
    num_patches    = 10000  # number of training examples
    max_iterations = 400    # number of optimization iterations

    visible_size = vis_patch_side ** 2  # number of input units
    hidden_size  = hid_patch_side ** 2  # number of hidden units
       
    training_data = loaddata(num_patches, vis_patch_side)
        
    encoder = SparseAutoEncoder(visible_size, hidden_size, rho, lamda, beta)
    
    opt_solution  = scipy.optimize.minimize(encoder.cost, encoder.theta, 
                                            args = (training_data,), method = 'L-BFGS-B', 
                                            jac = True, options = {'maxiter': max_iterations})
    opt_theta     = opt_solution.x
    opt_W1        = opt_theta[: hidden_size * visible_size].reshape(hidden_size, visible_size)

    visualizeW(opt_W1, vis_patch_side, hid_patch_side)

executeSparseAutoencoder()
