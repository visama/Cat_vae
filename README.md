# Categorical-Vae
Implementation of VAE for multivariate categorical data

Uses library https://github.com/HIPS/autograd for computing gradients of VAE loss function.

I'm still updating the project.

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

X_train should be in one-hot representation. Var_size should be a list containing number of possible events for each of the categorical variable in the training set. For Car-data [https://archive.ics.uci.edu/ml/datasets/car+evaluation] var_size is [4, 4, 4, 3, 3, 3, 4]. Car data contains 6 variables and one label with 4 possible values({'acc', 'good', 'unacc', 'vgood'}).

Vae object contains method "compress", which can be used for encoding data to a latent dimension z. By varying beta we can see how smaller beta enables latent dimension distribution to differ from standard normal distribution.

```python
z=vae_olio.compress(X_train)
```

<img src="plots/carvae_many_beta.png" width="600">
