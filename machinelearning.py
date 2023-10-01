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


# Interface for cost function
class ICost:

    def cost(self, actuals, expecteds):
        pass

    def derivative(self, actuals, expecteds):
        pass


# Implementation of ICost
class SquaredError(ICost):

    def cost(self, actuals, expecteds):
        sum = 0
        for (actual, expected) in zip(actuals, expecteds):
            sum += (actual - expected)**2
        return 0.5*sum

    def derivative(self, actuals, expecteds):
        derivatives = [a - e for (a, e) in zip(actuals, expecteds)]
        return np.array(derivatives)


# Class for tracking weights and biases information
# We need to store numOut x numIn weights and numOut biases
# Maybe have a set_error method and default error is 0
# Initialize weights and biases rando
# Store weights in a matrix. Maybe include bias
# Consider injectingn dependency for weight and bias initialization
# Make sure activation function applies componentwise to vectors
class Synapse:

    def __init__(self, numIn, numOut, activation):
        self.numIn = numIn
        self.numOut = numOut
        self.activation = activation
        # probably need to change this to be centered at 0
        rng = np.random.default_rng()
        self.weights = 2*rng.random((numOut, numIn)) - 1
        self.biases = rng.random((numOut, 1))

    def feedforward(self, x):
        self.x = x
        self.z = np.matmul(self.weights, x) + self.biases
        return self.activation.eval(self.z)

    def backprop(self, dActivations):
        dz = np.multiply(self.activation.derivative_at(self.z), dActivations)
        dWeights = np.matmul(dz, self.x)
        self.weights -= dWeights
        self.biases -= dz
        return np.matmul(dWeights.transpose(), dz)


# Class for neural network
class NeuralNet:

    def __init__(self, neuralLayers, activation, cost):
        self.synapses = []
        for i in range(1, len(neuralLayers)):
            prevLayer = neuralLayers[i - 1]
            thisLayer = neuralLayers[i]
            self.synapses.append(Synapse(prevLayer, thisLayer, Sigmoid()))
        self.cost = cost

    def feedforward(self, x):
        a = [x]
        for s in self.synapses:
            a.append(s.feedforward(a[-1]))
        self.output = a[-1]
        return a[-1]

    def get_cost(self, expecteds):
        return self.cost.cost(self.output, expecteds)

    def backprop(self, dActivations):
        a = [dActivations]
        for s in self.synapses:
            a.append(s.backprop(a[-1]))
