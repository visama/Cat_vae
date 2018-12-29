# Categorical-Vae
Implementation of VAE for multivariate categorical data

Uses library https://github.com/HIPS/autograd for computing gradients of VAE loss function.

I'm still updating the project.

Variational autoencoder is unsupervised method, that uses neural networks (NN) for estimating probability distributions. If data is X (nxd) we can use VAE for finding P(X). First NN is called encoder and the second decoder. Encoder encodes X to a latent dimension z and decoder builds a reconstruction X_r from z. Distribution for z is usually assumed to be multivariate standard normal P(z)=N(mu,I) and encoder maps X to the parameters mu and I of Q(z|X)=N(mu,I). Distribution for reconstruction X_r is product of categorical distributions P(X_r|z)=Prod(Cat(p_i)), where i=1...d. Vae can be trained using mini-batch stochastic gradient descent and the loss function is E[log(P(X_r|z))] - D(Q(z|X)||P(z)), where D is Kullback-Leibler divergence.

Example use:
```python
from Cat_vae.Vae_object import Vae
n,d = X_train.shape
T=500
q=0.09
s=0.01
layer_size=15
z_dim=2
encoder_layer_size=[d,layer_size,int(2*z_dim)] 
decoder_layer_size=[z_dim,layer_size,d]
vae_olio = Vae(encoder_layer_size,decoder_layer_size,var_size)
vae_olio.fit(X_train,T,q,s,beta=1,mcmc=2)
```

X_train should be in one-hot representation. Var_size should be a list containing number of possible events for each of the categorical variable in the training set. For Car-data [https://archive.ics.uci.edu/ml/datasets/car+evaluation], which I will be using as an example data, var_size is [4, 4, 4, 3, 3, 3, 4]. Car data contains 6 variables and one label with 4 possible values({'acc', 'good', 'unacc', 'vgood'}). T is the number of parameter updated, q is proportion of data points used in mini-batch SGD and s is the step size in mini-batch SGD. Layer size is the number of nodes in NN and z_dim is dimensionality of latent dimension. Beta and mcmc are optional. Beta controls importance of Kullback-Leibler divergence in loss function [https://openreview.net/references/pdf?id=Sy2fzU9gl] and mcmc is the number of mcmc samples used for calculation of E[log(P(X_r|z))].

VAE() creates a Vae object and initializes weights for boths NNs. Fit-method updates the weights with weights, that minimize the loss function. Vae object contains method "compress", which can be used for encoding data to a latent dimension z. By varying beta we can see how smaller beta enables latent dimension distribution to differ from standard normal distribution.

```python
z=vae_olio.compress(X_train)
```

<img src="plots/carvae_many_beta.png" width="600">
