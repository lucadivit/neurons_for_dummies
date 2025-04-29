import abc

class Activation(abc.ABC):

    @abc.abstractmethod
    def __call__(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

class Heaviside(Activation):

    def __call__(self, x) -> int:
        result = 1
        if x < 0:
            result = 0
        return result

    def __repr__(self) -> str:
        return self.__class__.__name__
