## Synthetic data generation tutorial 

Method sample creates a sample X_s from the estimated distribution. We could use X_s to fit a classification model and use it for predicting labels in the test set.

```python
from sklearn.linear_model import LogisticRegression
X_s=vae_olio.sample(n) # X_s sample size equals training data sample size

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train[:,0:21], y_train)
clf_z = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_s[:,0:21], z_train) # z_train should be vector containing the labels instead of one-hot matrix
print("Normal training set used, accuracy: "+str(clf.score(X_test[:,0:21],y_test)))
print("Synthetic training data used, accuracy: "+str(clf_z.score(X_test[:,0:21],y_test)))
Normal training data used, accuracy: 0.9176882661996497
Synthetic data used, accuracy: 0.8598949211908932
```

Based on this simple test synthetic data is somewhat usefull and conserves the structure of the original data! Perhaps by doing some hyperparameter tuning synthetic data quality could be improved. We could also inspect synthetic data quality visually by plotting the correlation matrices of both cases.

<img src="plots/Car correlation matrices.png" width="600">

Synthetic data correlation matrix looks quite authentic although slightly too granural.
