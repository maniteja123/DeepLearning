import numpy as np
from scipy import signal

def collect_speech_data():
    pass

def concatenate_speech_data(data, length):
    pass

class CDBN(object):

    def __init__(self, window_size, num_bases, spacing, pbias, pbias_lb, pbias_lambda, epsilon, l2reg, epsdecay, num_components, sigmaPC, K_CD, C_sigm, std_gaussian):
        self.windows_size = window_size
        self.num_bases = num_bases
        self.spacing = spacing
        self.pbias = pbias
        self.pbias_lb = pbias_lb
        self.pbias_lambda = pbias_lambda
        self.epsilon = epsilon
        self.l2reg = l2reg
        self.epsdecay = epsdecay
        self.num_channels = num_components
        self.sigmaPC = sigmaPC
        self.K_CD = K_CD
        self.C_sigm = C_sigm
        self.std_gaussian = std_guassian

    def whiten(self, X):
        num_samples, num_features = X.shape
        E = V[:, range(numfeatures, numfeat-nPC+1, -1)]
        S = diag(diag(D)[range(numfeatures, numfeat-nPC+1, -1)])
        Xpc = np.dot(E.T, X)
        Xrec = np.dot(E, Xpc)

        Ewhiten = np.dot(diag(sqrt(diag(S)+sigmaPC)**-1), E.T)
        Eunwhiten = np.dot(E, diag(sqrt(diag(S)+sigmaPC)))
        Xrec2 = np.dot(Eunwhiten, np.dot(Ewhiten, X))

        Xw = np.dot(Ewhiten, X)

    def train(self, X):
        num_frames = X.shape[0]
        X = self.trim_audio_for_spacing_fixconv(X)
        X = X.T
        X = X.reshape(X.shape[0], 1, X.shape[1])
        y = labels
        (ferr, dW, dh, dv, poshidprobs, poshidstates, negdata, stat)= self.sparse_conv(X, y)

    def sparse_conv(X, y):
        W = 0.01*randn(self.window_size, self.num_channels, self.num_bases)
        vis_bias = np.zeros(self.num_channels)
        hid_bias = -0.1 * np.ones(self.num_bases)

        poshidexp = tirbm_inference_fixconv_1d(X, W, hid_bias)
        poshidstates, poshidprobs = tirbm_sample_multrand_1d(poshidexp)
        
        posprods = tirbm_vishidprod_fixconv_1d(X, poshidprobs)

        poshidact = np.sum(np.sum(poshidprobs, 0), 1)
        posvisact = np.sum(np.sum(X, 0), 1)

        neghidstates = poshidstates
        for k in range(self.K_CD):
            negdata = tirbm_reconstruct_LB_fixconv_1d(neghidstates, W, pars)
            neghidexp = tirbm_inference_fixconv_1d(negdata, W, hid_bias)
            neghidstates, neghidprobs = tirbm_sample_multrand_1d(neghidexp)

            negprods = tirbm_vishidprod_fixconv_1d(negdata, neghidprobs)
            neghidact = np.sum(np.sum(neghidprobs, 0), 1)
            negvisact = np.sum(np.sum(negdata, 0), 1)

            fname_mat = sprintf("/home/student/mani/SPEECH/crbm_audio/results/audio/trainn/data%d.mat", labels)
            save(fname_mat, 'sizes', 'labels' , 'poshidexp', 'poshidact', 'posvisact', 'posprods', 'neghidexp', 'negprods','neghidact', 'negvisact')
            print("data saved as %s\n", fname_mat)

        ferr = np.mean((imdata - negdata) ** 2)

        dhbias = np.mean(np.mean(poshidprobs, 0), 1) - self.pbias
        dvbias = 0
        dW = 0

        numcases1 = poshidprobs.shape(0) * poshidprobs.shape(1)
        numcases2 = X.shape(0) * X.shape(1)

        dW_total1 = (posprods-negprods) / numcases1
        dW_total2 = - self.l2reg * W
        dW_total3 = - self.pbias_lambda * dW
        dW_total = dW_total1 + dW_total2 + dW_total3
        
        dWnorm_CD = np.linalg.norm(dW_total1)
        dWnorm_l2 = np.linalg.norm(dW_total2)
        stats = (dWnorm_CD, dWnorm_l2)

        dh_total = (poshidact-neghidact) / numcases1 - self.pbias_lambda * dhbias
        dv_total = (posvisact-negvisact) / numcases2

        print("||W||=%g, ||dWprod|| = %g, ||dWl2|| = %g, ||dWsparse|| = %g\n", np.sqrt(np.sum(W ** 2)), np.sqrt(np.sum(dW_total1 ** 2)), np.sqrt(np.sum(dW_total2 ** 2)), np.sqrt(np.sum(dW_total3 ** 2)))

    def trim_audio_for_spacing_fixconv(self, X):
        ws = self.window_size
        spacing = self.spacing
        if X.shape[1]-ws+1 % spacing != 0:
            n = X.shape[1]-ws+1 % spacing
            X[:, 1:floor(n/2), :] = []
            X[:, X.shape[1]-ceil(n/2)+1:, :] = []
        return X

    def tirbm_inference_fixconv_1d(self, X, W, hid_bias):
        num_frames = X.shape[0]
        poshidexp = np.zeros(num_frames-self.window_size+1, 1, self.num_bases)
        poshidprobs = np.zeros_like(poshidexp)
    
        for c in range(self.num_channels):
            H = W[range(W.shape[0], 1, -1), c, :]
            H = H.reshape(self.window_size, 1, self.num_bases)
            poshid = self.conv2_mult(X[:,:,c], H, "valid")
            poshidexp = poshidexp + poshid.reshape(poshidexp.shape)

        for b in range(self.num_bases):
            poshidexp[:,:,b] = self.C_sigm/(self.std_gaussian**2) * (poshidexp2[:,:,b] + hid_bias[b])
            poshidprobs[:,:,b] = 1./(1 + np.exp(-poshidexp2[:,:,b]))

        return (poshidexp, poshidprob)

    def tirbm_reconstruct_LB_fixconv_1d(self, S, W):
        num_frames = S.shape[0]
        negdata = np.zeros(num_frames + self.window_size-1, 1, self.num_channels)

        for b in range(self.num_bases):
            H = W[:,:,b].reshape(ws,1,numchannels)
            negdata = negdata + self.conv2_mult(S1[:,:,b], H, "full")

        negdata = self.C_sigm * negdata

        return negdata

    def tirbm_sample_multrand_1d(self, poshidexp):
        poshidexp = np.max(np.min(poshidexp,20), -20) # DEBUG: This is for preventing NAN values
        poshidprobs = np.exp(poshidexp)
        poshidprobs_mult = np.zeros(self.spacing+1, poshidprobs.shape[0]*poshidprobs.shape[1]*poshidprobs.shape[2]/spacing)
        poshidprobs_mult[-1, :] = 1

        for r in range(self.spacing):
            temp = poshidprobs[range(r, poshidprobs.shape[0], shaping), :, :]
            poshidprobs_mult[r,:] = temp[:]
            [S, P] = multrand2(poshidprobs_mult.T)
            S = S.T
            P = P.T

        H = np.zeros_like(poshidexp)
        HP = np.zeros_like(poshidexp)
        for r in range(self.spacing):
            Sr = S[r,:]
            H[range(r, H.shape[0], spacing), :, :] = Sr.reshape(H.shape[0]/spacing, H.shape[1], H.shape[2])
            Pr = P[r,:]
            HP[range(r, H.shape[0], spacing), :, :] = Pr.reshape(H.shape[0]/spacing, H.shape[1], H.shape[2])

        Sc = np.sum(S[:S.shape[0], :])
        Pc = np.sum(P[:P.shape[0], :])
        Hc = Sc.reshape(poshidexp.shape[0]/spacing, poshidexp.shape[1], poshidexp.shape[2])
        HPc = Pc.reshape(poshidexp.shape[0]/spacing, poshidexp.shape[1], poshidexp.shape[2])

        return (poshidpexp, poshidprob)

    def tirbm_vishidprod_fixconv_1d(self, X, H):

        selidx = range(H.shape[0], 0, -1)
        vishidprod = np.zeros(self.window_size, 1, self.num_channels, self.num_bases)

        for b in range(self.num_bases):
            vishidprod[:,:,:,b] = self.conv2_mult2(X, H[selidx, :, b], "valid")

        vishidprod = vishidprod.reshape(self.window_size, self.num_channels, self.num_bases)
    
        return vishidprod

    def conv2_mult(self, a, B, convopt):
        y = np.zeros_like(B)
        for i in range(B.shape[2]):
            y[:,:,i] = signal.convolve(a, B[:,:,i], convopt)
        return y

    def conv2_mult2(self, A, b, convopt):
        y = np.zeros_like(A)
        for i in range(A.shape[2]):
            y[:,:,i] = signal.convolve(A[:,:,i], b, convopt)
        return y

    def multrand2(seld, P):
        P = np.mean(P, 1)
        cumP = np.cumsum(P, 1)
        unifrnd = rand(P.shape[0], 1)
        temp = cumP > unifrnd
        Sindx = diff(temp, 1, 2)
        S = np.zeros_like(P)
        S[:, 0] = 1 - np.sum(Sindx, 1)
        S[:, 1:] = Sindx
