from Cat_vae.misc_vae import addition, nro_params_in_NN, Loss_function
from Cat_vae.optimizers import adam
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc import flatten
import random

def cat_vae(X,var_size,encoder_params,decoder_params,T,q,s,beta,mcmc):
    
    def objective(params_tuple):
        encoder_params, decoder_params = params_tuple
        return -Loss_function(encoder_params, decoder_params, x,var_size,beta,mcmc)
    
    #calculate nro of parameters
    nro_paramsEnc=nro_params_in_NN(encoder_params)
    nro_paramsDec=nro_params_in_NN(decoder_params)
    
    #m=q*n, where m is the subsample size used in the stochastic gradient descent
    n,d=X.shape
    m=int(q*len(X))

    #initialize for Adam-optimizer
    me=np.zeros(nro_paramsEnc)
    ve=np.zeros(nro_paramsEnc)
    md=np.zeros(nro_paramsDec)
    vd=np.zeros(nro_paramsDec)
    
    #monitoring of losses 
    losses = []
    
    for t in range(T):
        objective_grad = grad(objective)
        subsample_ind=random.sample(range(len(X)),m)
        losss = 0.0
        x=X[subsample_ind]
        grad_enc, grad_dec = objective_grad((encoder_params,decoder_params))
        gradientti_enc, unflatten_enc  = flatten(grad_enc)
        gradientti_dec, unflatten_dec = flatten(grad_dec)
            
        #losss = losss + Loss_function(encoder_params, decoder_params, x,var_size,beta,mcmc) /m
        #losses.append(losss) 
        
        #Update encoder
        update,me,ve = adam(gradientti_enc,me,ve,t,step_size=s)
        encoder_params = addition(encoder_params, unflatten_enc(update))
        
        #Update decoder
        update,md,vd = adam(gradientti_dec,md,vd,t,step_size=s)
        decoder_params = addition(decoder_params, unflatten_dec(update))
   
        #if t % 10 == 0:
            #print(losses[t])
    return([encoder_params,decoder_params])
