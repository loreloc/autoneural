import numpy
import keras
import scipy
import statistics
from ensemble import Ensemble

# The classifier ensemble
class EnsembleClassifier(Ensemble):
    # Initialize the ensemble with a list of classifier models
    def __init__(self, models):
        super().__init__(models)

    # Compute the prediction of the ensemble as the mode of the predictions
    # of each classifier model
    def predict(self, x):
        categories = list(map(lambda m:
            list(map(lambda c:numpy.argmax(c), m.predict(x))), self.models))
        mode, _ = scipy.stats.mode(categories)
        return mode[0]

    # Evaluate the ensemble using the categorical accuracy metric
    def evaluate(self, x_test, y_test):
        y_hat = self.predict(x_test)
        y_test = list(map(lambda c:numpy.argmax(c), y_test))
        return statistics.mean(numpy.equal(y_test, y_hat).astype(float))
