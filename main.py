import machinelearning as ml
import numpy as np
import matplotlib.pyplot as plt


def main():
    nl = [1, 5, 1]
    sig = ml.Sigmoid()
    uniform = ml.Normal(0, 2)
    cost = ml.SquaredError()
    lr = 0.1
    n = 200
    epochs = 500
    nn = ml.NeuralNet(nl, [sig], uniform, cost, lr)
    rng = np.random.default_rng()
    xs = rng.uniform(0, 2*np.pi, n)
    training_set = [(np.array([[x]]), np.array([[np.sin(x)]])) for x in xs]
    nn.train(training_set, epochs)
    tnum = 100
    xtest = np.linspace(0, 2*np.pi, tnum)
    ytest = np.array([nn.feedforward(np.array([[x]]))[0] for x in xtest])
    yp = np.sin(xtest)
    plt.plot(xtest, yp, "r", label="y=f(x)")
    plt.scatter(xtest, ytest, marker=".", label="Model")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
