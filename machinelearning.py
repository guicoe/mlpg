import abc
import numpy as np


# Interface for activation functions
# eval must be able to apply componentwise to vector x
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


# Class for tracking weights and biases information
# We need to store numOut x numIn weights and numOut biases
# Maybe have a set_error method and default error is 0
# Initialize weights and biases rando
# Store weights in a matrix. Maybe include bias
# Consider injectingn dependency for weight and bias initialization
# Make sure activation function applies componentwise to vectors
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


# Interface for cost function
class ICost:

    def cost(self, actuals, expecteds):
        pass

    def derivative(self, actuals, expecteds):
        pass


# Implementation of ICost
class SquaredError(ICost):

    def cost(self, actual, expected):
        return 0.5*np.sum(np.multiply(actual - expected, actual - expected))

    def derivative(self, actual, expected):
        return actual - expected


# Class for neural network
class NeuralNet:

    def __init__(self, neuralLayers, activations, cost, stepSize):
        self.synapses = []
        rng = np.random.default_rng()
        for x, y in zip(neuralLayers, neuralLayers[1:]):
            weights = rng.normal(0, 10, size=(y, x))
            biases = -1*np.ones((y, 1))
            self.synapses.append(LinearSynapse(weights, biases, stepSize))
            if activations:
                self.synapses.append(activations.pop(0))
        self.cost = cost

    def feedforward(self, x):
        for s in self.synapses:
            x = s.feedforward(x)
        self.output = x
        return self.output


    def backprop(self, expecteds):
        dy = self.cost.derivative(self.output, expecteds)
        for s in reversed(self.synapses):
            dy = s.backprop(dy)
