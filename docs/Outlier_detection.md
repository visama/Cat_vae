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

Based on the plots above it is clear, that outlier data points have indeed a smaller likelihood and reconstruction probabilities. Next we could try to find out what is the proportion of outlier data points within n data points, that get the smallest probabilities. Here n  is the size of the outlier data set.

```python
#Join inlier and outlier probabilities and sort them
all_probs_rec = np.concatenate((I_rec_probs,O_rec_probs))
all_probs_like = np.concatenate((I_likelihood_probs,O_likelihood_probs))

sorted_probs_rec = np.sort(all_probs_rec)
sorted_probs_like = np.sort(all_probs_like)

n = X_O.shape[0]
print("% of outliers among smallest rec-prob points: " + str(sum(O_rec_probs < sorted_probs_rec[n])/X_O.shape[0]))
print("% of outliers among smallest likelihood-prob points: " + str(sum(O_likelihood_probs < sorted_probs_like[n])/X_O.shape[0]))
print("% of outliers among all points points: " + str(X_O.shape[0]/(X_I.shape[0]+X_O.shape[0])))
print("Size of all data points: " +str(X_O.shape[0]+X_I.shape[0]))
print("Size of outlier data set: " +str(X_O.shape[0]))
% of outliers among smallest rec-prob points: 1.0
% of outliers among smallest likelihood-prob points: 0.9357541899441341
% of outliers among all points points: 0.38536060279870826
Size of all data points: 929
Size of outlier data set: 358
```
Nearly all of the outliers are among the data points with smallest probabilities! Reconstruction probabilities seems to be working a bit better here, than likelihood probabilities.

But what does it mean for a multivariate categorical data point to be an outlier? We could select the data point given smallest and highest probability and find out. Remember, that the variable names are "buying price", "maintenance price", "number of doors", "capacity", "luggage boot size" and "safety". The label variable is "acceptability" of the car.

```python
print("Lowest inlier: "+inverse_x_1hot(X_I[list(I_rec_probs).index(min(I_rec_probs))],var_size,data))
print("Lowest outlier: "+inverse_x_1hot(X_O[list(O_rec_probs).index(min(O_rec_probs))],var_size,data))
print("Highest inlier: "+inverse_x_1hot(X_I[list(I_rec_probs).index(max(I_rec_probs))],var_size,data))
print("Highest outlier: "+inverse_x_1hot(X_O[list(O_rec_probs).index(max(O_rec_probs))],var_size,data))
Lowest inlier: ['low', 'high', '3', 'more', 'med', 'high', 'vgood']
Lowest outlier: ['low', 'vhigh', '2', '2', 'med', 'low', 'vgood']
Highest inlier: ['low', 'low', '5more', '4', 'big', 'high', 'vgood']
Highest outlier: ['med', 'med', '5more', '4', 'med', 'high', 'acc']
```

## One class as outlier-class


## References
[1] http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf
[2] https://arxiv.org/pdf/1802.03903.pdf
