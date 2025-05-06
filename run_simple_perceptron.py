from Perceptron import Perceptron
from Activation import Heaviside
import numpy as np
from colorama import Fore, Back, Style

p = Perceptron(dimension=2, activation=Heaviside())
p.weights = np.array([0.5, 0.5])
p.bias = 1.

cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
true = [0, 0, 0, 1]
for i, (case, y) in enumerate(zip(cases, true)):
    prediction = p(inputs=np.array(case))
    if prediction == y:
        print(Fore.GREEN + f"Case {i + 1}. Input = {case}, True Result = {y}, Predicted Result = {prediction}")
    else:
        print(Fore.RED + f"Case {i + 1}. Input = {case}, True Result = {y}, Predicted Result = {prediction}")