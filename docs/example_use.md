## Example use - fitting Vae-object and compressing data with encoder

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

X_train should be numpy.ndarray in one-hot representation. Var_size should be a list containing number of possible events for each of the categorical variables in the training set. For Car-data [https://archive.ics.uci.edu/ml/datasets/car+evaluation], which I will be using as an example data, var_size is [4, 4, 4, 3, 3, 3, 4]. Car data contains 6 variables and one label with 4 possible values({'acc', 'good', 'unacc', 'vgood'}). 

T is the number of times NN parameters are updated, q is proportion of data points used in mini-batch SGD and s is the step size in mini-batch SGD. Layer size is the number of nodes in NN and z_dim is dimensionality of latent dimension. Here NNs have only one hidden layer, but also multilayer NNs are possible. Beta and mcmc are optional and are by default one. Beta one corresponds original VAE. Beta controls importance of Kullback-Leibler divergence in loss function [https://openreview.net/references/pdf?id=Sy2fzU9gl] and mcmc is the number of mcmc samples used for calculation of E[log(P(X_r|z))].

VAE() creates a Vae object and initializes weights for boths NNs. Fit-method updates the weights with weights, that minimize the loss function. Vae object contains method compress, which can be used for encoding data to a latent dimension z. By varying beta we can see from the plot below, how smaller beta enables latent dimension distribution to differ from standard normal distribution. Encoder maps data points of the same class close to each other. If one wants to use VAE for synthetic data generation beta should be one. This is because we can create synthetic data by feeding the decoder data from multivariate normal distribution, since decoder is learned to map this kind of data.

```python
z=vae_olio.compress(X_train)
```

<img src="plots/carvae_many_beta.png" width="600">
