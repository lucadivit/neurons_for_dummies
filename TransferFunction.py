import abc, numpy as np


class TransferFunction(abc.ABC):

    def __call__(self, inputs: np.ndarray, weights: np.ndarray, bias: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def derive(self, x: float, w: float, b: float) -> (float, float):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

class WeightedSum(TransferFunction):

    def __call__(self, inputs: np.ndarray, weights: np.ndarray, bias: float) -> float:
        return np.dot(inputs, weights) + bias

    def derive(self, x: float, w: float, b: float) -> (float, float):
        return x, 1.

    def __repr__(self) -> str:
        return self.__class__.__name__