import abc, numpy as np


class Error(abc.ABC):

    @abc.abstractmethod
    def __call__(self, true: float, pred: float) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def derive(self, true: float, pred: float) -> float:
        raise NotImplementedError

class BinaryCrossEntropy(Error):

    def __clip_prediction(self, pred: float) -> float:
        threshold = 1e-10
        # Avoid 0 or 1
        pred = np.clip(pred, threshold, 1 - threshold)
        return pred

    def __check_values(self, true: float, pred: float) -> (float, float):
        assert true in [0., 1.], "true must be 0 or 1"
        assert 0 <= pred <= 1, "pred must be between 0 and 1"
        return true, pred

    def __call__(self, true: float, pred: float) -> float:
        true, pred = self.__check_values(true, pred)
        pred = self.__clip_prediction(pred)
        result = - (true * np.log(pred) + (1 - true) * np.log(1 - pred))
        result = float(round(result, 5))
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__

    def derive(self, true: float, pred: float) -> float:
        true, pred = self.__check_values(true, pred)
        pred = self.__clip_prediction(pred)
        result = - ( true / pred - ( (1 - true) / (1 - pred) ) )
        result = float(round(result, 5))
        return result

