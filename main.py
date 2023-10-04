import machinelearning as ml
import numpy as np
import matplotlib.pyplot as plt


def main():
    sig = ml.Sigmoid()
    nn = ml.NeuralNet([1, 5, 1], sig, ml.SquaredError())
    rng = np.random.default_rng()
    n = 200
    xs = rng.uniform(-1, 1, n)
    ys = 2*xs**2 + 1
    epochs = 8000
    for _ in range(epochs):
        for x, y in zip(xs, ys):
            nn.feedforward(x)
            nn.backprop(y)
    tnum = 100
    xtest = np.linspace(-1, 1, tnum)
    ytest = np.array([nn.feedforward(x)[0] for x in xtest])
    yp = 2*xtest**2 + 1
    plt.plot(xtest, yp, "r", label="y=f(x)")
    plt.scatter(xtest, ytest, marker=".", label="Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
