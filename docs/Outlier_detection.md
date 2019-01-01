# Outlier detection tutorial
I will follow the example of [1] and [2] and use Vae for detecting outlier of Car-data set. First approach uses reconstruction errors and second one likelihoods in the task. We get a reconstruction probability for a data point x simple by feeding it to the encoder and the reconstructing it with the decoder. Then we can calculate the probabilty multiple times and take the average as our final answer. Likelihood is achieved just like reconstruction probability, but this time we feed decoder with random numbers from the standard normal distribution.
## Random data points as outliers
Car data has 6912 possible unique data points, but the sample size of the data is only 1728. We could just randomly pick a possible data point, that is not included in the data and assign it as an outlier. Original data would be split to training and inlier sets. We will train the model as usual with the training set and calculate likelihood and reconstruction probabilities for inliers and outliers. We would assume, that outlier probabilities differ from inlier ones by being smaller.


```python
###Method reconstruction_prob and likelihood can be used to calculate the needed probabilities
O_rec_probs = vae_olio.reconstruction_prob(X_O,500)
I_rec_probs = vae_olio.reconstruction_prob(X_I,500)

O_likelihood_probs = vae_olio.likelihood(X_O,500)
I_likelihood_probs = vae_olio.likelihood(X_I,500)
```
<img src="plots/outlier histograms.png" width="600">



## One class as outlier-class


## References
[1] http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
[2] https://arxiv.org/pdf/1802.03903.pdf
