from Perceptron import Perceptron
from Activation import Heaviside
import numpy as np

p = Perceptron(dimension=2, activation=Heaviside())
p.weights = np.array([0.5, 0.5])
p.bias = 1.

cases = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i, case in enumerate(cases):
    prediction = p(inputs=np.array(case))
    print(f"Case {i + 1}. Input = {case}, Prediction = {prediction}")