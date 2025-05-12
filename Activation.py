import abc, numpy as np

class NotDifferentiableError(Exception):
    def __init__(self, function_name: str):
        super().__init__(f"Function '{function_name}' is not differentiable.")

class Activation(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def derive(self, x) -> float|int:
        raise NotImplementedError

class Heaviside(Activation):

    def __call__(self, x: float) -> int:
        result = 1
        if x < 0:
            result = 0
        return result

    def derive(self, x) -> int:
        raise NotDifferentiableError(self.__class__.__name__)

    def __repr__(self) -> str:
        return self.__class__.__name__

class Sigmoid(Activation):

    def __call__(self, x: float) -> float:
        threshold = 700
        x = np.clip(x, -threshold, threshold)
        result = (1 / (1 + np.exp(-x))).item()
        result = float(round(result, 5))
        return result

    def derive(self, x) -> float:
        return self(x=x) * (1 - self(x=x))

    def __repr__(self) -> str:
        return self.__class__.__name__