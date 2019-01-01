# Outlier detection tutorial

## Random data points as outliers
Car data has 6912 possible unique data points, but the sample size of the data is only 1728. We could just randomly pick a possible data point, that is not included in the data and assign it as an outlier. Original data would be split to training and inlier sets. We will train the model as usual with the training set and calculate likelihood and reconstruction probabilities for inliers and outliers. We would assume, that outlier probabilities differ from inlier ones by being smaller.


```python
###Methoda reconstruction_prob and likelihood can be used to calculate the needed probabilities
O_rec_probs = vae_olio.reconstruction_prob(X_O,500)
I_rec_probs = vae_olio.reconstruction_prob(X_I,500)

O_likelihood_probs = vae_olio.likelihood(X_O,500)
I_likelihood_probs = vae_olio.likelihood(X_I,500)
```
<img src="plots/outlier histograms.png" width="600">



## One class as outlier-class


## References
