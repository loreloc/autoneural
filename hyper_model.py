import math
import keras
import bayes_opt
import statistics

# The available activation functions hyperparameters
Activations = ['relu', 'softplus', 'sigmoid', 'tanh']

# A general, multi-purpose, hyper-model:
# An hyper-model is composed by an input and output size,
# a fixed depth (number of hidden layers) and
# a maximum width (number of units for each layer)
class HyperModel:
    # Constructor for the hyper-model
    def __init__(self, input_size, output_size, depth, width, activation,
            loss, metrics
        ):
        self.input_size = input_size
        self.output_size = output_size
        self.depth = depth
        self.width = width
        self.activation = activation
        self.loss = loss
        self.metrics = metrics

    # Get the best model found
    def best(self, x_train, y_train, x_val, y_val, iter):
        # Get the optimizer result
        optimizer = self.optimize(x_train, y_train, x_val, y_val, iter)
        # Compile the best model
        return self.compile(optimizer.max['params'])

    # Get an ensemble of the best `k` models found
    def ensemble(self, x_train, y_train, x_val, y_val, iter, k):
        # Get the optimizer result
        optimizer = self.optimize(x_train, y_train, x_val, y_val, iter)
        # Sort the dictionaries of paramethers based on the target value
        results = sorted(optimizer.res, key=lambda d:d['target'], reverse=True)
        # Take the first `k` paramethers dictionaries
        params = list(map(lambda d:d['params'], results))[:k]
        # Compile each model of the ensemble
        models = list(map(lambda p:self.compile(p), params))
        return models

    # Optimize the hyper-model for a specified task and return the optimizer
    def optimize(self, x_train, y_train, x_val, y_val, iter):
        # The black-box function to optimize
        # (e.g. find the minimum log-loss on the validation set)
        def hyper_model_validate(**params):
            # Build the model and optimizer
            model, optimizer = self.build(params)
            # Compile the model
            model.compile(optimizer, self.loss, None)
            # Fit the model
            model.fit(x_train, y_train, verbose=0)
            # Calculate the log-loss value
            return -math.log2(model.evaluate(x_val, y_val, verbose=0))

        # Prepare the bounds of the Gaussian Process
        pbounds = {
            'lrate': (1e-4, 1e-2),   # The learning rate
            'beta1': (0.80, 0.9900), # The first paramether of the Adam optimizer
            'beta2': (0.99, 0.9999)  # The second paramether of the Adam optimizer
        }

        for i in range(self.depth):
            pbounds['width' + str(i)] = (math.sqrt(self.width), self.width)
            pbounds['activ' + str(i)] = (0.0, len(Activations) - 1)
            pbounds['regul' + str(i)] = (0.0, 1e-3)

        # Instantiate a BayesianOptimization object
        # using the GP bounds and the function to maximize
        optimizer = bayes_opt.BayesianOptimization(
            f=hyper_model_validate, pbounds=pbounds, verbose=2
        )

        # Maximize the black-box function (using Expected Improvement)
        optimizer.maximize(init_points=10, n_iter=iter, acq='ei')
        return optimizer

    # Compile a model from the hyper-model
    def compile(self, params):
        # Build the hyper-model
        model, optimizer = self.build(params)
        # Compile the model
        model.compile(optimizer, self.loss, self.metrics)
        return model

    # Build a model and an optimizer from the hyper-model
    def build(self, params):
        # Instantiate a (Keras) MLP model
        model = keras.models.Sequential()

        # Populate the MLP with specific regularized layers
        input_size = self.input_size
        for i in range(0, self.depth):
            width = int(round(params['width' + str(i)]))
            a_idx = int(round(params['activ' + str(i)]))
            regul = params['regul' + str(i)]
            model.add(keras.layers.Dense(
                width,
                activation=Activations[a_idx],
                kernel_regularizer=keras.regularizers.l2(regul),
                input_shape=(input_size,)
            ))
            input_size = width

        # Add the output layer to the MLP
        model.add(keras.layers.Dense(
            self.output_size,
            activation=self.activation,
            input_shape=(input_size,)
        ))

        # Instantiate the optimizer
        optimizer = keras.optimizers.Adam(
            params['lrate'], params['beta1'], params['beta2']
        )

        return model, optimizer
