# autoneural
**autoneural** is a simple library based on **Keras** used to experiment with hyper-parameters automatic tuning for neural networks using Bayesian optimization. At the moment the target model is a feed-forward dense neural network. The following hyper-parameters are optimized:
- Learning rate
- Layers width
- Layers activation functions
- Layers L2 regularizer constants
- Parameters used by the Adam optimizer

## MNIST Digits Classification example
Run the following command:
```
$ python mnist.py
```

## Boston Housing Prices Regression example
Run the following command:
```
$ python boston.py
```

