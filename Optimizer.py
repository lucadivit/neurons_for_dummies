import abc, numpy as np
from copy import deepcopy

from Error import Error
from Perceptron import Perceptron
from colorama import Fore, Style


class Optimizer(abc.ABC):

    def __init__(self, lr: float, error_function: Error):
        self._lr = lr
        self._error = error_function

    @property
    def error(self) -> Error:
        return deepcopy(self._error)

    @property
    def lr(self) -> float:
        return self._lr

    @lr.setter
    def lr(self, lr: float):
        self._lr = lr

    @error.setter
    def error(self, error_function: Error):
        self._error = error_function

    @abc.abstractmethod
    def __call__(self, perceptron: Perceptron, inputs: list, outputs: list, verbose: bool = False) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

class GradientDescent(Optimizer):

    def __call__(self, perceptron: Perceptron, inputs: list, outputs: list, verbose: bool = False) -> float:
        error = 0
        for i, (case, y) in enumerate(zip(inputs, outputs)):
            case = np.array(case)

            net_output = perceptron(inputs=case, call_activation=False)
            prediction = perceptron.activation(x=net_output)

            weights = perceptron.weights
            bias = perceptron.bias

            dL_dpred = self._error.derive(true=y, pred=prediction)
            dpred_dz = perceptron.activation.derive(x=net_output)
            dL_dz = dL_dpred * dpred_dz

            if verbose:
                print(Fore.YELLOW + f"---------------------- Train on Case {case} = {y} ----------------------")
                print(Fore.MAGENTA + "dL/dpred =", dL_dpred)
                print(Fore.MAGENTA + "dpred/dz =", dpred_dz)
                print(Fore.MAGENTA + f"dL_dz = dL/dpred * dpred/dz = {dL_dpred} * {dpred_dz} = {dL_dz}")

            dL_dwi = []
            dL_dbi = []
            if type(bias) in (int, float):
                bias = [float(bias) for _ in range(len(weights))]

            for i, (w, x, b) in enumerate(zip(weights, case, bias)):

                dz_dw, dz_db = perceptron.transfer_function.derive(x=x, w=w, b=b)

                dL_dw = dL_dz * dz_dw
                dL_dwi.append(dL_dw)

                dL_db = dL_dz * dz_db
                dL_dbi.append(dL_db)

                if verbose:
                    print(Fore.MAGENTA + f"dL_dw{i} = dL_dz * dz_dw{i} = dL/dpred * dpred/dz * dz_dw{i} = {dL_dpred} * {dpred_dz} * {dz_dw} = {dL_dw}")
                    print(Fore.MAGENTA + f"dL_db{i} = dL_dz * dz_db{i} = dL/dpred * dpred/dz * dz_db{i} = {dL_dpred} * {dpred_dz} * {dz_db} = {dL_db}")

            # In case of weighted sum transfer function, bias derivative is always 1 hence dL_dbi has always the same value.
            if len(set(dL_dbi)) != 1:
                raise Exception(Fore.RED + "Cannot handle bias as a function of parameters")

            bias = bias[0]
            dL_db = dL_dbi[0]
            new_bias = float(bias - self._lr * dL_db)

            new_weights = []
            for i, (w, dL_dw) in enumerate(zip(weights, dL_dwi)):
                new_w = float(w - self._lr * dL_dw)
                new_weights.append(new_w)
                if verbose:
                    print(Fore.MAGENTA + f"w{i} = w{i} - lr * dL_dw{i} = {w} - {self._lr} * {dL_dw} = {new_w}")

            new_weights = np.array(new_weights)

            if verbose:
                print(Fore.MAGENTA + f"b = b - lr * dL_db = {bias} - {self._lr} * {dL_db} = {new_bias}")
                print(Fore.MAGENTA + f"Old Weights = {perceptron.weights}, Old Bias = {perceptron.bias}")
                print(Fore.MAGENTA + f"New Weights = {new_weights}, New Bias = {new_bias}")

            perceptron.weights = new_weights
            perceptron.bias = new_bias
            error += self._error(true=y, pred=perceptron(inputs=case))

        print(Style.RESET_ALL)
        return error

    def __repr__(self) -> str:
        return self.__class__.__name__