import scipy.io
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from numpy import random
import pickle
import timeit
from sklearn.model_selection import GridSearchCV

import snips as snp  # my snippets

"""*Disclaimer: This little exercise with this movie ratings toy data set is very common. You can see two examples [here](http://blog.ethanrosenthal.com/2015/11/02/intro-to-collaborative-filtering/) and [here](http://online.cambridgecoding.com/notebooks/mhaller/implementing-your-own-recommender-systems-in-python-using-stochastic-gradient-descent-4) of people doing _exactly_ what I'm about to do.*"""

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)
df.head()

"""We expect most users haven't rated most movies, so let's get a sense of exactly how sparse the data is. The number of missing ratings should be the difference between the number of rows here, and the total number of possible ratings $n_{users} \times n_{movies}$."""

n_u = len(df["user_id"].unique())
n_m = len(df["item_id"].unique())
sparsity = len(df) / (n_u * n_m)
print("sparsity of ratings is %.2f%%" % (sparsity * 100))

# Split the dataframe into a train and test set
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(df, test_size=0.25)

train_data = pd.DataFrame(train_data)
test_data = pd.DataFrame(test_data)

# Create training and test matrix
R = np.zeros((n_u, n_m))
for line in train_data.itertuples():
    R[line[1] - 1, line[2] - 1] = line[3]

T = np.zeros((n_u, n_m))
for line in test_data.itertuples():
    T[line[1] - 1, line[2] - 1] = line[3]

"""# Implementation with Stochastic Gradient Descent
Cambridge Coding Academy Online has a neat tutorial where they implement this exact algorithm with this exact Toy Data Set. They chose to use Stochastic Gradient Descent for their optimzation, which loops through individual ratings (making several full passes through the training set) and updates the relevant latent factors each time. This approch can't be matricized as effectively as normal gradient descent, but it tends to converge more quickly so in terms of full computational cost it's not clear which is better (might depend on the size of the data set). For more on the difference between these two approaches, [here is a neat and brief Quora answer](https://www.quora.com/Whats-the-difference-between-gradient-descent-and-stochastic-gradient-descent/answer/Sebastian-Raschka-1?srid=vrN4). I'm going to try implementing it both ways.
"""

start_time = timeit.default_timer()


# Scoring Function: Root Mean Squared Error
def rmse_score(R, Q, P):
    I = R != 0  # Indicator function which is zero for missing data
    ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
    MSE = ME ** 2
    return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors


# Set parameters and initialize latent factors
f = 20  # Number of latent factor pairs
lmbda = 0.5  # Regularisation strength
gamma = 0.01  # Learning rate
n_epochs = 50  # Number of loops through training data
P = 3 * np.random.rand(n_u, f)  # Latent factors for users
Q = 3 * np.random.rand(n_m, f)  # Latent factors for movies

# Stochastic GD
train_errors = []
test_errors = []
users, items = R.nonzero()
for epoch in range(n_epochs):
    for u, i in zip(users, items):
        e = R[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
        P[u, :] += gamma * (e * Q[i, :] - lmbda * P[u, :])  # Update this user's features
        Q[i, :] += gamma * (e * P[u, :] - lmbda * Q[i, :])  # Update this movie's features
    train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass
    test_errors.append(rmse_score(T, Q, P))  # Test RMSE for this pass

# Print how long it took
print("Run took %.2f seconds" % (timeit.default_timer() - start_time))

# Check performance by plotting train and test errors
fig, ax = plt.subplots()
ax.plot(train_errors, color="g", label='Training RMSE')
ax.plot(test_errors, color="r", label='Test RMSE')
snp.labs("Number of Epochs", "RMSE", "Error During Stochastic GD")
ax.legend()

# See how well we did on Test Set Predictions
Rhat = np.dot(P, Q.T)
fig, axs = plt.subplots(figsize=[5, 10], nrows=5, ncols=1, sharex=True)
fig.suptitle("Stochastic GD Test Performance")
for idx, ax in enumerate(axs.ravel()):
    vals = Rhat[T == idx + 1]
    ax.hist(vals, bins=20, normed=True, label="Ground Truth Rating = %i" % (idx + 1))
    ax.legend()
    ax.set_xlim([0, 6])

"""# Implementation with Batch Gradient Descent"""

start_time = timeit.default_timer()


# Scoring Function: Root Mean Squared Error
def rmse_score(R, Q, P):
    I = R != 0  # Indicator function which is zero for missing data
    ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
    MSE = ME ** 2
    return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors


# Set parameters and initialize latent factors
f = 20  # Number of latent factor pairs
lmbda = 50  # Regularisation strength
gamma = 9e-5  # Learning rate
n_epochs = 220  # Number of loops through training data
P = 3 * np.random.rand(n_u, f)  # Latent factors for users
Q = 3 * np.random.rand(n_m, f)  # Latent factors for movies

# Batch GD
train_errors = []
test_errors = []
for epoch in range(n_epochs):
    ERR = np.multiply(R != 0, R - np.dot(P, Q.T))  # compute error with present values of Q, P, ZERO if no rating   
    #     P += gamma*(np.dot(Q.T, ERR.T).T - lmbda*P)  # update rule
    #     Q += gamma*(np.dot(P.T, ERR).T - lmbda*Q)  # update rule

    P += gamma * (np.dot(ERR, Q) - lmbda * P)  # update rule
    Q += gamma * (np.dot(ERR.T, P) - lmbda * Q)  # update rule

    train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass
    test_errors.append(rmse_score(T, Q, P))  # Test RMSE for this pass

# Print how long it took
print("Run took %.2f seconds" % (timeit.default_timer() - start_time))

# Check performance by plotting train and test errors
fig, ax = plt.subplots()
ax.plot(train_errors, color="g", label='Training RMSE')
ax.plot(test_errors, color="r", label='Test RMSE')
snp.labs("Number of Epochs", "RMSE", "Error During Batch GD")
ax.legend()

# See how well we did on Test Set Predictions
Rhat = np.dot(P, Q.T)
fig, axs = plt.subplots(figsize=[5, 10], nrows=5, ncols=1, sharex=True)
fig.suptitle("Batch GD Test Performance")
for idx, ax in enumerate(axs.ravel()):
    vals = Rhat[T == idx + 1]
    ax.hist(vals, bins=20, normed=True, label="Ground Truth Rating = %i" % (idx + 1))
    ax.legend()
    ax.set_xlim([0, 6])

"""# Rolling a Custom Estimator for Sklearn
I love `GridSearchCV` and `Pipeline` so I'd really like to wrap the above algorithm in sklearn-API-compatible way. I'm referring to [the official dev docs](http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator) on this topic. A great paper from 2013 on the general structure and philosophy of sklearn can he had [on ArXiv](https://arxiv.org/pdf/1309.0238v1.pdf). I'm going to leave both batch and stochastic GD as options by giving my class a parameter `solver` where we can specify which. 

So without further ado, let me grab the relevant dev helper objects, like the base class `BaseEstimator` that I will inherit from.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class MC(BaseEstimator):
    """ An estimator for latent factor collaborative filtering models in Recommender Systems.
    """

    def __init__(self, n_u, n_m, n_factors=10, n_epochs=250, lmbda=10, gamma=9e-5, solver="sgd"):
        self.n_u = n_u
        self.n_m = n_m
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lmbda = lmbda
        self.gamma = gamma
        self.solver = solver

    def fit(self, X, y):
        """Fits all the latent factors for users and items and saves the resulting matrix representations.
        """
        X, y = check_X_y(X, y)

        # Create training matrix
        R = np.zeros((self.n_u, self.n_m))
        for idx, row in enumerate(X):
            R[row[0] - 1, row[1] - 1] = y[idx]

            # Initialize latent factors
        P = 3 * np.random.rand(self.n_u, self.n_factors)  # Latent factors for users
        Q = 3 * np.random.rand(self.n_m, self.n_factors)  # Latent factors for movies

        def rmse_score(R, Q, P):
            I = R != 0  # Indicator function which is zero for missing data
            ME = I * (R - np.dot(P, Q.T))  # Errors between real and predicted ratings
            MSE = ME ** 2
            return np.sqrt(np.sum(MSE) / np.sum(I))  # sum of squared errors

        # Fit with stochastic or batch gradient descent
        train_errors = []
        if self.solver == "sgd":
            # Stochastic GD
            users, items = R.nonzero()
            for epoch in range(self.n_epochs):
                for u, i in zip(users, items):
                    e = R[u, i] - np.dot(P[u, :], Q[i, :].T)  # Error for this observation
                    P[u, :] += self.gamma * (e * Q[i, :] - self.lmbda * P[u, :])  # Update this user's features
                    Q[i, :] += self.gamma * (e * P[u, :] - self.lmbda * Q[i, :])  # Update this movie's features
                train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass
        elif self.solver == "batch_gd":
            # Batch GD
            for epoch in range(self.n_epochs):
                ERR = np.multiply(R != 0,
                                  R - np.dot(P, Q.T))  # compute error with present values of Q, P, ZERO if no rating
                P += self.gamma * (np.dot(Q.T, ERR.T).T - self.lmbda * P)  # update rule
                Q += self.gamma * (np.dot(P.T, ERR).T - self.lmbda * Q)  # update rule
                train_errors.append(rmse_score(R, Q, P))  # Training RMSE for this pass
        else:
            print("I'm sorry, we don't recognize that solver.")

        #         print("Completed %i epochs, final RMSE = %.2f" %(self.n_epochs, train_errors[-1]))
        self.Q = Q
        self.P = P
        self.train_errors = train_errors

        # Return the estimator
        return self

    def predict(self, X):
        """ Predicts a vector of ratings from a matrix of user and item ids.
        """
        X = check_array(X)

        y = np.zeros(len(X))
        PRED = np.dot(self.P, self.Q.T)
        for idx, row in enumerate(X):
            y[idx] = PRED[row[0] - 1, row[1] - 1]

        return y

    def score(self, X, y):
        """ Element-wise root mean squared error.
        """
        yp = self.predict(X)
        err = y - yp
        mse = np.sum(np.multiply(err, err)) / len(err)
        return np.sqrt(mse)


"""OK now that the class is defined let's use the full data set to get $X$ and $y$ in the proper format. `GridSearchCV` will do it's own train/test splitting with K-Folds. I also designed the class to require the total number of users and movies as parameters in the constructor, that was just the simplest thing to do, so we need to count those and then pass them in."""

X = df[["user_id", "item_id"]].as_matrix()
y = df["rating"].values
n_u = len(df["user_id"].unique())
n_m = len(df["item_id"].unique())

"""### Grid Search with Batch GD"""

rcmdr = MC(n_u=n_u, n_m=n_m, gamma=6e-5, n_epochs=400, solver="batch_gd")
params = {"lmbda": (45, 50, 55),
          "n_factors": (15, 18, 21)}
search = GridSearchCV(rcmdr, param_grid=params, cv=4)
search.fit(X, y)

best_est = search.best_estimator_
results = pd.DataFrame(search.cv_results_)
results[["mean_test_score", "std_test_score", "params"]].sort_values(by=["mean_test_score"], ascending=True).head()

"""### Grid Search with Stochastic GD"""

rcmdr = MC(n_u=n_u, n_m=n_m, gamma=0.01, n_epochs=50, solver="sgd")
params = {"lmbda": (0.25, 0.5, 0.75),
          "n_factors": (18, 20, 22)}
search = GridSearchCV(rcmdr, param_grid=params, cv=4)
search.fit(X, y)

best_est = search.best_estimator_
results = pd.DataFrame(search.cv_results_)
results[["mean_test_score", "std_test_score", "params"]].sort_values(by=["mean_test_score"], ascending=True).head()
