from Perceptron import Perceptron
from Activation import Sigmoid
from Optimizer import GradientDescent
from Error import BinaryCrossEntropy
from Metric import Accuracy
import numpy as np

optimizer = GradientDescent(lr=0.5, error_function=BinaryCrossEntropy())

perceptron = Perceptron(dimension=2, activation=Sigmoid())
perceptron.weights = np.array([0.5, 0.5])
perceptron.bias = 1.

cases = [[0, 1], [1, 0], [0, 0], [1, 1]]
true = [0, 0, 0, 1]

metric = Accuracy()
def compute_metric() -> float:
    predictions = []
    for case in cases:
        prediction = perceptron(inputs=np.array(case))
        predictions.append(0 if prediction < 0.5 else 1)
    return metric(true=np.array(true), pred=np.array(predictions))

epochs = 3
for epoch in range(epochs):
    print(f"Start Epoch {epoch + 1}/{epochs}")
    print()
    error = optimizer(perceptron=perceptron, inputs=cases, outputs=true, verbose=True)
    print(f"End Epoch {epoch + 1}/{epochs}. Error: {error}, Accuracy: {compute_metric()}%")
    print()
