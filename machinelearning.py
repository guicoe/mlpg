import abc
import numpy as np


class IInitializer(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'initialize') and
                callable(subclass.initialize) or
                NotImplemented)

    @abc.abstractmethod
    def initialize(self, size):
        raise NotImplementedError


class Normal(IInitializer):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def initialize(self, size):
        rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=size)


class Uniform(IInitializer):

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def initialize(self, size):
        rng = np.random.default_rng()
        return rng.uniform(self.min, self.max, size=size)

        
class ISynapse(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'feedforward') and
                callable(subclass.feedforward) and
                hasattr(subclass, 'backprop') and
                callable(subclass.backprop) or
                NotImplemented)

    @abc.abstractmethod
    def feedforward(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def backprop(self, dy):
        raise NotImplementedError


# Class for sigmoid implementation of activation function
class Sigmoid(ISynapse):

    def feedforward(self, x):
        self.z = 1 / (1 + np.exp(-x))
        return self.z

    def backprop(self, dy):
        dz = np.multiply(self.z, 1 - self.z)
        return np.multiply(dz, dy)


class Identity(ISynapse):

    def feedforward(self, x):
        return x

    def backprop(self, dy):
        return dy


class LinearSynapse(ISynapse):

    def __init__(self, weights, biases, stepSize):
        self.weights = weights
        self.biases = biases
        self.stepSize = stepSize

    def feedforward(self, x):
        self.x = x
        return self.weights@self.x + self.biases

    def backprop(self, dy):
        dWeights = dy@self.x.transpose()
        self.weights -= self.stepSize*dWeights
        self.biases -= self.stepSize*dy
        return dWeights.transpose()@dy


class ICost:

    def cost(self, actuals, expecteds):
        pass

    def derivative(self, actuals, expecteds):
        pass


class SquaredError(ICost):

    def cost(self, actual, expected):
        return 0.5*np.sum(np.multiply(actual - expected, actual - expected))

    def derivative(self, actual, expected):
        return actual - expected


class NeuralNet:

    def __init__(self, neuralLayers, activations, initializer, cost, stepSize):
        self.build_synapses(neuralLayers, activations, initializer, stepSize)
        self.cost = cost

    def build_synapses(self, neuralLayers, activations, initializer, stepSize):
        self.synapses = []
        for n, m in zip(neuralLayers, neuralLayers[1:]):
            weights = initializer.initialize((m, n))
            biases = 1*np.ones((m, 1))
            self.synapses.append(LinearSynapse(weights, biases, stepSize))
            if activations:
                self.synapses.append(activations.pop(0))

    def cycle(self, x, y):
        self.feedforward(x)
        self.backprop(y)

    def train_epoch(self, training_set):
        for x, y in training_set:
            self.cycle(x, y)

    def train(self, training_set, epochs):
        for _ in range(epochs):
            self.train_epoch(training_set)

    def feedforward(self, x):
        for s in self.synapses:
            x = s.feedforward(x)
        self.output = x
        return self.output

    def backprop(self, expecteds):
        dy = self.cost.derivative(self.output, expecteds)
        for s in reversed(self.synapses):
            dy = s.backprop(dy)
