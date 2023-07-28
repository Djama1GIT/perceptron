import sys

import numpy as np

sys.setrecursionlimit(2000000)


def foo(a, b, c, d, e, f):
    return a and b and c or d and e or f


class Perceptron:
    def __init__(self, _inputs, _outputs):
        self.inputs = _inputs
        self.outputs = _outputs
        self.synapses = np.random.randn(_inputs.shape[1], _outputs.shape[1])

    def __call__(self, *args, **kwargs):
        self.backpropagation()
        return self.synapses

    def backpropagation(self):
        for i in range(1000000):
            __inputs = self.inputs
            __outputs = self.ReLU(np.dot(__inputs, self.synapses))

            err = self.outputs - __outputs
            adjustments = np.dot(__inputs.T, err * self.ReLU_derivative(__outputs))

            self.synapses += adjustments

    @staticmethod
    def ReLU(x: np.array) -> np.array:
        return np.maximum(x, 0)

    @staticmethod
    def ReLU_derivative(x: np.array) -> np.array:
        return np.where(x > 0, 1, 0)


class MLP:
    def __init__(self, inputs, outputs, hidden_units=None):
        self.inputs = inputs
        self.outputs = outputs
        self.hidden_units = hidden_units or [10, 10]
        self.synapses = []
        for i in range(len(self.hidden_units) + 1):
            if i == 0:
                synapse = np.random.randn(inputs.shape[1], self.hidden_units[i])
            elif i == len(self.hidden_units):
                synapse = np.random.randn(self.hidden_units[-1], outputs.shape[1])
            else:
                synapse = np.random.randn(self.hidden_units[i - 1], self.hidden_units[i])
            self.synapses.append(synapse)

    def __call__(self, *args, **kwargs):
        self.backpropagation()
        return self.synapses

    def backpropagation(self):
        for i in range(1000):
            inputs = self.inputs
            layers = [inputs]
            for j in range(len(self.hidden_units)):
                layer = np.tanh(np.dot(layers[-1], self.synapses[j]))
                layers.append(layer)
            outputs = np.tanh(np.dot(layers[-1], self.synapses[-1]))

            output_error = self.outputs - outputs
            output_adjustments = np.dot(layers[-1].T, output_error * (1 - np.square(outputs)))

            hidden_errors = [output_error]
            hidden_adjustments = []
            for j in range(len(self.hidden_units), 0, -1):
                error = np.dot(hidden_errors[-1], self.synapses[j].T)
                hidden_errors.append(error)
                adjustment = np.dot(layers[j - 1].T, error * (1 - np.square(layers[j])))
                hidden_adjustments.append(adjustment)
            hidden_adjustments.reverse()

            self.synapses[-1] += output_adjustments
            for j in range(len(self.hidden_units)):
                self.synapses[j] += hidden_adjustments[j]


counter = 0
mlp: MLP = None


def test_perceptron():
    global mlp
    inputs = np.array([
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1],
        [0, 1, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [1, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 0],
        [1, 1, 1, 0, 0, 1],
    ])

    outputs = np.array([[foo(*i)] for i in inputs])

    mlp = MLP(inputs, outputs, hidden_units=[2, 3])
    synapses = mlp()

    new_inputs = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 0],
    ])

    expected_outputs = np.array([[foo(*i)] for i in new_inputs])
    global counter
    for i in range(len(new_inputs)):
        result = np.tanh(np.dot(np.tanh(np.dot(new_inputs[i], synapses[0])), synapses[1]))
        result = np.round(result, decimals=2)
        print(f"Input: {new_inputs[i]} Output: {result[0]}, Expected: {expected_outputs[i][0]}")
        if (result[0] > 0) != bool(expected_outputs[i][0]):
            counter += 1
            print(f"Тест #{counter} не пройден.")
            test_perceptron()
            break


test_perceptron()
print(f"Тест #{counter} пройден.")
print(mlp.synapses)
