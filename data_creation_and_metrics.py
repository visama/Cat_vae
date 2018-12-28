# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 15:26:06 2018

@author: visam
"""
import numpy as np
from Cat_vae.misc_vae import neural_network,unpack_gaussian_params2,create_z2

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def categorical_sample(p,n):
    return(np.random.multinomial(1, p, size=n))
    
def categorical_multisample(ps,n):
    Xs=[]
    for p in ps:
        Xs.append(np.random.multinomial(1, p, size=n))
    return(np.hstack(Xs))
    
def reconstruction_cat_vae(params_D,n,z_dim,var_size):
    gaus_noise=np.random.randn(n,z_dim)
    probs_for_rec=[]
    vali2=neural_network(params_D,gaus_noise)
    for i in range(n):
        vali = vali2[i]
        length=len(var_size)
        paikka=0
        ttt=[]
        for j in range(length):
            ttt.append(softmax(vali[paikka:int(paikka+var_size[j])]))
            paikka = paikka + var_size[j]
    
        jotain=[]
        for jjj in ttt:
            for kkk in jjj:
                jotain.append(kkk)
        probs_for_rec.append(jotain)
    np.asarray(probs_for_rec)
#sample reconstruction
    reconstruction=[]
    for i in range(n):
        probab=probs_for_rec[i]
        length=len(var_size)
        paikka=0
        ttt=[]
        for j in range(length):
            ttt.append(categorical_sample(probab[paikka:int(paikka+var_size[j])],1)[0])
            paikka = paikka + var_size[j]
        jotain=[]
        ttt=ttt
        for jjj in ttt:
            for kkk in jjj:
                jotain.append(kkk)
        reconstruction.append(jotain)    
    reconstruction=np.array(reconstruction)
    return(reconstruction)
  
