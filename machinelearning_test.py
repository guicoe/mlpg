from unittest.mock import patch
import unittest
import numpy as np
import machinelearning as ml


class TestUniform(unittest.TestCase):

    def test_initialize(self):
        sut = ml.Uniform(-2, 9)
        a = sut.min
        b = sut.max
        self.assertGreaterEqual(a, -2)
        self.assertLessEqual(b, 9)


class TestLinearSynapse(unittest.TestCase):

    def setUp(self):
        weights = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
        biases = np.array([7, 8]).reshape(2, 1)
        self.sut = ml.LinearSynapse(weights, biases, 1)

    def test_feedforward_shape(self):
        x = np.array([3, 0, 5]).reshape(3, 1)
        result = self.sut.feedforward(x)
        self.assertTupleEqual(result.shape, (2, 1))

    def test_feedforward_value(self):
        x = np.array([3, 0, 5]).reshape(3, 1)
        result = self.sut.feedforward(x).flatten().tolist()
        self.assertListEqual(result, [25, 50])

    @patch.object(ml.LinearSynapse, 'update')
    def test_backprop(self, mock_update):
        self.sut.x = np.zeros((3, 1))
        dy = np.array([7, 2]).reshape(2,1)
        result = self.sut.backprop(dy).flatten().tolist()
        self.assertListEqual(result, [15, 24, 33])


if __name__ == "__main__":
    unittest.main()
