import machinelearning as ml
import numpy as np


def main():
    sig = ml.Sigmoid()
    nn = ml.NeuralNet([1, 5, 1], sig, ml.SquaredError())
    rng = np.random.default_rng()
    xs = [2*rng.random((1, 1)) - 1 for _ in range(200)]
    ys = [x**2 for x in xs]
    for i in range(200):
        nn.feedforward(xs[i])
        nn.get_cost(ys[i])
    tnum = 5
    #test = [nn.feedforward(np.array([[-1 + dx]])) for dx in range(0, 2, 2/tnum)]
    xtest = np.linspace(-1.0, 1.0, tnum)
    print(xtest)
    ytest = [nn.feedforward(np.reshape(x, (1, 1))) for x in xtest]
    print(ytest)


if __name__ == "__main__":
    main()
