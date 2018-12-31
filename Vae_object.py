# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:42:45 2018

@author: visam
"""
from Cat_vae.misc_vae import encode, neural_network
from Cat_vae.misc_vae import init_net_params_xavier_2nd as init_net_params, Reconstruction_prob_multi
from Cat_vae.categorical_vae import cat_vae
from Cat_vae.data_creation_and_metrics import reconstruction_cat_vae
import autograd.numpy.random as npr

class Vae:
    def __init__(self,encoder_layer_size,decoder_layer_size,var_size):
        self.params_E = init_net_params(0.1, encoder_layer_size, rs=npr.RandomState(0))
        self.params_D = init_net_params(0.1, decoder_layer_size, rs=npr.RandomState(0))
        self.encoder_layer_size = encoder_layer_size
        self.decoder_layer_size = decoder_layer_size
        self.var_size = var_size

    def fit(self,X,T,q,s,beta=1,mcmc=1):
        self.params_E,self.params_D = cat_vae(X,self.var_size,self.params_E,self.params_D,T,q,s,beta,mcmc)
        
    def compress(self,X):
        return(encode(self.params_E,X))
        
    def sample(self,n):
        return(reconstruction_cat_vae(self.params_D,n,self.decoder_layer_size[0],self.var_size))
        
    def reconstruction_prob(self,X,n):
        return(Reconstruction_prob_multi(self.params_E, self.params_D,self.var_size,X,n))
     
