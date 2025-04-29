import numpy as np
from Activation import Activation

class Perceptron:

    def __init__(self, dimension: int, activation: Activation = None):
        assert dimension >= 2, "Dimension must be >= 2"
        self._weights = np.zeros(dimension)
        self._bias = 0.
        self._activation = activation

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def bias(self) -> float:
        return self._bias

    @weights.setter
    def weights(self, weights: np.ndarray):
        dim = self._weights.shape[0]
        assert len(weights.shape) == 1 and weights.shape[0] == dim, f"Weights must be a vector with shape ({dim}, )"
        self._weights = weights

    @bias.setter
    def bias(self, bias: float):
        self._bias = bias

    def __call__(self, inputs: np.ndarray) -> float:
        dim = self._weights.shape[0]
        assert len(inputs.shape) == 1 and inputs.shape[0] == dim, f"Inputs must be a vector with shape ({dim}, )"
        weighted_sum = np.dot(inputs, self._weights) + self._bias
        result = weighted_sum
        if self._activation is not None:
            result = self._activation(weighted_sum)
        return result

    def __repr__(self) -> str:
        weights = ""
        for i, w in enumerate(self._weights):
            if i + 1 == len(self._weights):
                weights += f"w_{i}={w}"
            else:
                weights += f"w_{i}={w}, "
        return f"Perceptron({weights}, bias={self._bias}, activation={repr(self._activation)})"