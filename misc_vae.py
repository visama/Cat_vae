import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
import matplotlib.pyplot as plt
import pandas as pd
from autograd import grad
from autograd.misc import flatten
import random
from autograd.scipy.misc import logsumexp

def init_net_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a (weights, biases) tuples for all layers."""
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_net_params_xavier(scale, layer_sizes, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n)*np.sqrt(1/m),   # weight matrix
             scale * rs.randn(n)*np.sqrt(1/m)) # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_net_params_xavier_2nd(scale, layer_sizes, rs=npr.RandomState(0)):
    return [(scale * rs.randn(m, n)*np.sqrt(1/(m+n)),   # weight matrix
             scale * rs.randn(n)*np.sqrt(1/(m+n))) # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
 
def unpack_gaussian_params2(params):
    # Params of a diagonal Gaussian.
    D = np.shape(params)[-1] // 2
    mean, log_std = params[:, :D], params[:, D:]
    return mean, log_std

def create_z2(mu,log_var):
    #sample from the normal distr
    eps = np.random.randn(mu.shape[0],mu.shape[1])
    #reparametrization trick
    z=mu + eps * np.exp(log_var)
    return(z)
    
def neural_network(params,inputs):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = np.tanh(outputs)
    return(outputs)

def encode(encoder_params,x):
    means, log_stds = unpack_gaussian_params2(neural_network(encoder_params, x))
    z = create_z2(means, log_stds)
    return(z)
    
def KL(myy,cov):
    return(0.5*np.sum(np.exp(cov) + myy**2 - 1 - cov))
    
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x),axis=1)[:,None]

def nro_params_in_NN(params):
    nro_params=0
    for iii in params:
        nro_params= nro_params+ np.size(iii[0]) + np.size(iii[1])
    return(nro_params)
    
def addition(a,b):
    flattened_a, unflatten = flatten(a)
    flattened_b, _ = flatten(b)
    return(unflatten(flattened_a+flattened_b))
    
#Decoder outputs real numbers, but we want probabilities, that sum up to one
#list var_size holds information about the number of events for each categorical variable
#for example var_size[0] = number of events for first variable 
    
#To achieve that I have sliced decoder output to d submatrices with sizes [m,var_size[1]],...,[m,var_size[d]]
#where m is the size of the subsample fed to SGD
    
#In the end I feed these matrices to softmax function which outputs the probabilities needed
def multivariate_categorical_loss(var_size,decoder_NN_output,x):
    res = 0
    paikka = 0
    for i in range(len(var_size)):
        probs = softmax(decoder_NN_output[:,paikka:paikka+var_size[i]]) #syötä funktioon myös one hot X?
        res = res + np.sum(np.log(probs)*x[:,paikka:paikka+var_size[i]])
        paikka = paikka + var_size[i]
    return(res)
    
def Loss_function(encoder_params, decoder_params, x,var_size,beta,mcmc_size):
    means, log_stds = unpack_gaussian_params2(neural_network(encoder_params, x))
    KL_div=KL(means,log_stds)
    z = create_z2(means, log_stds)
    decoder_NN_output=neural_network(decoder_params, z)
    
    res = 0
    for i in range(mcmc_size):
        res = res + multivariate_categorical_loss(var_size,decoder_NN_output,x)/mcmc_size
    
    return(res  - beta*KL_div)

def Reconstruction_prob(encoder_params, decoder_params,var_size,x): # x is matrix of samples
    means, log_stds = unpack_gaussian_params2(neural_network(encoder_params, x))
    z = create_z2(means, log_stds)
    decoder_NN_output=neural_network(decoder_params, z)
    res = np.zeros(x.shape[0])
    paikka = 0
    for i in range(len(var_size)):
        probs = softmax(decoder_NN_output[:,paikka:paikka+var_size[i]])
        res = res + np.sum(np.log(probs)*x[:,paikka:paikka+var_size[i]],axis=1)
        paikka = paikka + var_size[i]
    return(res)

def Reconstruction_prob_multi(encoder_params, decoder_params,var_size,x,nro_samples):
    res = np.zeros(x.shape[0])
    for i in range(nro_samples):
        res = res + Reconstruction_prob(encoder_params, decoder_params,var_size,x)/nro_samples
    return(res)
        
