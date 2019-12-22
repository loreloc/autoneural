import keras
from hyper_model import HyperModel
from ensemble_classifier import EnsembleClassifier

# A general, hyper-classifier
class HyperClassifier(HyperModel):
    Loss = 'categorical_crossentropy'
    Metric = 'categorical_accuracy'

    # Constructor for the hyper-classifier
    def __init__(self, input_size, output_size, depth, width):
        super().__init__(input_size, output_size, depth, width, 'softmax',
            self.Loss, [self.Metric])

    # Get the best classifier found
    def best(self, x_train, y_train, x_val, y_val, iter=10):
        return super().best(x_train, y_train, x_val, y_val, iter)

    # Get an ensemble classifier
    def ensemble(self, x_train, y_train, x_val, y_val, iter=10, k=5):
        # Get the best models found
        models = super().ensemble(x_train, y_train, x_val, y_val, iter, k)
        return EnsembleClassifier(models)
