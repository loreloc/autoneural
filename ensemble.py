# The ensemble model class
class Ensemble:
    # Initialize the ensemble with a list of compiled models
    def __init__(self, models):
        self.models = models

    # Fit each model in the ensemble
    def fit(self, x_train, y_train, epochs=10):
        for n, model in enumerate(self.models):
            print("Fitting model #" + str(n))
            model.fit(x_train, y_train, epochs=epochs, verbose=0)
