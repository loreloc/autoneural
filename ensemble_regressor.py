import keras
import statistics
from ensemble import Ensemble

# The regressor ensemble
class EnsembleRegressor(Ensemble):
    # Initialize the ensemble with a list of regressor models
    def __init__(self, models):
        super().__init__(models)

    # Compute the prediction of the ensemble as the average of the predictions
    # of each regressor model
    def predict(self, x):
        s = sum(list(map(lambda m:m.predict(x), self.models))).flatten()
        return s / len(self.models)

    # Evaluate the ensemble using the mean absolute error metric
    def evaluate(self, x_test, y_test):
        y_hat = self.predict(x_test)
        return statistics.mean(abs(y_test - y_hat))
