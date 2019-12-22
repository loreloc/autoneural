import keras
from hyper_model import HyperModel
from ensemble_regressor import EnsembleRegressor

# A general, hyper-regressor
class HyperRegressor(HyperModel):
    Loss = 'mean_squared_error'
    Metric = 'mean_absolute_error'

    # Constructor for the hyper-regressor
    def __init__(self, input_size, output_size, depth, width):
        super().__init__(input_size, output_size, depth, width, 'linear',
            self.Loss, [self.Metric])

    # Get the best regressor found
    def best(self, x_train, y_train, x_val, y_val, iter=10):
        return super().best(x_train, y_train, x_val, y_val, iter)

    # Get an ensemble regressor
    def ensemble(self, x_train, y_train, x_val, y_val, iter=10, k=5):
        # Get the best models found
        models = super().ensemble(x_train, y_train, x_val, y_val, iter, k)
        return EnsembleRegressor(models)
