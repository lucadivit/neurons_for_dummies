import numpy as np
from Activation import Activation
from TransferFunction import TransferFunction, WeightedSum
from copy import deepcopy

class Perceptron:

    def __init__(self, dimension: int, activation: Activation = None, transfer_function: TransferFunction = WeightedSum()):
        assert dimension >= 2, "Dimension must be >= 2"
        self._weights = np.zeros(dimension)
        self._bias = 0.
        self._activation = activation
        self._transfer_function = transfer_function

    @property
    def weights(self) -> np.ndarray:
        return deepcopy(self._weights)

    @property
    def bias(self) -> float:
        return self._bias

    @property
    def activation(self) -> Activation:
        return deepcopy(self._activation)

    @property
    def transfer_function(self) -> TransferFunction:
        return deepcopy(self._transfer_function)

    @weights.setter
    def weights(self, weights: np.ndarray):
        dim = self._weights.shape[0]
        assert len(weights.shape) == 1 and weights.shape[0] == dim, f"Weights must be a vector with shape ({dim}, )"
        self._weights = weights

    @bias.setter
    def bias(self, bias: float):
        assert type(bias) == float, "Bias must be a float"
        self._bias = bias

    @activation.setter
    def activation(self, activation: Activation):
        self._activation = activation

    @transfer_function.setter
    def transfer_function(self, transfer_function: TransferFunction):
        self._transfer_function = transfer_function

    def __call__(self, inputs: np.ndarray, call_activation: bool = True) -> float:
        dim = self._weights.shape[0]
        assert len(inputs.shape) == 1 and inputs.shape[0] == dim, f"Inputs must be a vector with shape ({dim}, )"
        result = self._transfer_function(inputs=inputs, weights=self._weights, bias=self._bias)
        if self._activation is not None and call_activation is True:
            result = self._activation(x=result)
        return result

    def __repr__(self) -> str:
        weights = ""
        for i, w in enumerate(self._weights):
            if i + 1 == len(self._weights):
                weights += f"w_{i}={w}"
            else:
                weights += f"w_{i}={w}, "
        return f"Perceptron({weights}, bias={self._bias}, activation={repr(self._activation)}, transfer_function={repr(self._transfer_function)})"