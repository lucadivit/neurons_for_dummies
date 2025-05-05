import abc, numpy as np

class Metric(abc.ABC):

    @abc.abstractmethod
    def __call__(self, true: np.ndarray, pred: np.ndarray) -> float:
        raise NotImplementedError

class Accuracy(Metric):
    def __call__(self, true: np.ndarray, pred: np.ndarray) -> float:
        if len(true) != len(pred):
            raise ValueError("true and pred must have the same length")
        correct = (true == pred).sum()
        result = correct / len(true) if len(true) > 0 else 0.0
        return float(round(result, 2))