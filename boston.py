import keras
import numpy as np
import sklearn.model_selection
from hyper_regressor import HyperRegressor

# Preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Construct the train set and the validation set randomly
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train, y_train
)

# Instantiate the hyper model
hyper_regressor = HyperRegressor(13, 1, depth=3, width=64)

# Get the ensemble regressor
regressor = hyper_regressor.ensemble(
    x_train, y_train, x_val, y_val, iter=100, k=10
)

# Fit the regressor
x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
regressor.fit(x_train, y_train, epochs=100)

# Evaluate the regressor
mae = regressor.evaluate(x_test, y_test)
print("Mean Absolute Error of the Ensemble: " + str(mae))
