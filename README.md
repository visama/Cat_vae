# Categorical-Vae
Implementation of VAE for multivariate categorical data

Uses library https://github.com/HIPS/autograd for computing gradients of VAE loss function.

I'm still updating the project.

Variational autoencoder is unsupervised method, that uses two neural networks (NN) for estimating probability distributions. If data is X (nxd) we can use VAE for finding P(X). First NN is called encoder and the second decoder. Encoder encodes X to a latent dimension z and decoder builds a reconstruction X_r from z. Distribution for z is usually assumed to be multivariate standard normal P(z)=N(mu,I) and encoder maps X to the parameters mu and I of Q(z|X)=N(mu,I). Distribution for reconstruction X_r is product of categorical distributions P(X_r|z)=Prod(Cat(p_i)), where i=1...d. Vae can be trained using mini-batch stochastic gradient descent and the loss function is E[log(P(X|z))] - D(Q(z|X)||P(z)), where D is Kullback-Leibler divergence. This implementation does not use sampling on the output layer of the decoder. Instead softmax output layer and cross entropy loss for each variable is used.

Vae is a versatile tool and it could be used in synthetic data generation, anomaly detection, missing data imputation Et cetera.


[Example use](docs/example_use.md)

[Synthetic data](docs/Synth_data.md)

[Outlier_detection](docs/Outlier_detection.md)




