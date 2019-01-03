## Vae-imputation of multivariate categorical data

Here we follow an example of [1] in our vae-imputation experiments. Idea is to again use
likelihood probabilities and consider a data point likely is it's attached with high probability. Algorithm for imputating one data point x is

1. Fit Vae using the training set X_train

2. Construct a histogram of likelihood probabilities of X_train

3. Now if H is the distribution of the probabilities, find h for which P(H<h)=alfa

4. Fill missing part of x using random sampling from categorical distribution with equal probabilities for each event

5. Calculate likelihood probability for new data point(x_i) and if x_i<h, choose x_i as imputated data point. Else go back to part 4. and fill x again using random sampling

Paper [1] doesn't mention using histogram of training set probabilities as a method for finding h, but idea is to reject data points that are deemed very unlikely by Vae. Alfa could perhaps be 0.05 or 0.1.


Refs
[1] John T. McCoy et al.: Variational Autoencoders for Missing Data Imputation with Application to a Simulated Milling Circuit
