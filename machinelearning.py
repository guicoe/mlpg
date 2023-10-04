import numpy as np


# Interface for activation functions
# eval must be able to apply componentwise to vector x
class IActivationFunction:

    def eval(self, x):
        pass

    def derivative_at(self, x):
        pass


# Class for sigmoid implementation of activation function
class Sigmoid(IActivationFunction):

    def eval(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative_at(self, x):
        return np.multiply(self.eval(x), 1 - self.eval(x))


class Identity(IActivationFunction):

    def eval(self, x):
        return x

    def derivative_at(self, x):
        return np.ones(x.shape)


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


# Class for tracking weights and biases information
# We need to store numOut x numIn weights and numOut biases
# Maybe have a set_error method and default error is 0
# Initialize weights and biases rando
# Store weights in a matrix. Maybe include bias
# Consider injectingn dependency for weight and bias initialization
# Make sure activation function applies componentwise to vectors
class Synapse:

    def __init__(self, numIn, numOut, activation, stepSize):
        self.numIn = numIn
        self.numOut = numOut
        self.stepSize = stepSize
        self.activation = activation
        rng = np.random.default_rng()
        self.weights = rng.normal(0, 5, size=(numOut, numIn))
        self.biases = -1*np.ones((numOut, 1))

    def feedforward(self, x):
        self.x = x.reshape(self.numIn, 1)
        self.z = self.weights@self.x + self.biases
        return self.activation.eval(self.z)

    def backprop(self, dActivations):
        dActivations = dActivations.reshape(self.numOut, 1)
        dz = np.multiply(self.activation.derivative_at(self.z), dActivations)
        dWeights = dz@self.x.transpose()
        self.weights -= self.stepSize*dWeights
        self.biases -= self.stepSize*dz
        return dWeights.transpose()@dz


# Class for neural network
class NeuralNet:

    def __init__(self, neuralLayers, activation, cost):
        self.synapses = []
        self.numOut = neuralLayers[-1]
        for i in range(len(neuralLayers) - 2):
            prevLayer = neuralLayers[i]
            thisLayer = neuralLayers[i + 1]
            self.synapses.append(Synapse(prevLayer, thisLayer, Sigmoid(), 0.01))
        self.synapses.append(Synapse(neuralLayers[-2], neuralLayers[-1], Identity(), 0.01))
        self.cost = cost

    def feedforward(self, x):
        a = [x]
        for s in self.synapses:
            a.append(s.feedforward(a[-1]))
        self.output = a[-1]
        return a[-1]

    def get_cost(self, expecteds):
        expecteds = expecteds.reshape(self.numOut, 1)
        return self.cost.cost(self.output, expecteds)

    def backprop(self, expecteds):
        a = [self.cost.derivative(self.output, expecteds)]
        for s in reversed(self.synapses):
            a.append(s.backprop(a[-1]))
