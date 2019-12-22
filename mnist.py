import keras
import numpy as np
import sklearn.model_selection
from hyper_classifier import HyperClassifier

# Preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
x_test = x_test.reshape(10000, 784).astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Construct the train set and the validation set randomly
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    x_train, y_train
)

# Instantiate the hyper model
hyper_classifier = HyperClassifier(784, 10, depth=4, width=1200)

# Get the ensemble classifier
classifier = hyper_classifier.ensemble(
    x_train, y_train, x_val, y_val, iter=100, k=10
)

# Fit the classifier
x_train = np.concatenate((x_train, x_val))
y_train = np.concatenate((y_train, y_val))
classifier.fit(x_train, y_train, epochs=20)

# Evaluate the classifier
acc = classifier.evaluate(x_test, y_test)
print("Categorical Accuracy of the Ensemble: " + str(acc))
