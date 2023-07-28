import numpy as np


class Perceptron:
    def __init__(self, _inputs, _outputs):
        self.inputs = _inputs
        self.outputs = _outputs
        self.synapses = 2 * np.random.random((len(self.inputs[0]), 1)) - 1

    def __call__(self, *args, **kwargs):
        self.backpropagation()
        return self.synapses

    def backpropagation(self):
        for i in range(20000):
            __input = self.inputs
            __outputs = Perceptron.sigmoid(np.dot(__input, self.synapses))

            err = self.outputs - __outputs
            adjustments = np.dot(__input.T, err * (__outputs * (1 - __outputs)))

            self.synapses += adjustments

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


inputs = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1],
])

outputs = np.array([
    [0],
    [1],
    [1],
    [0],
])

perceptron = Perceptron(inputs, outputs)
print(f"[Synapses]:\n>>> " + '\n>>> '.join(map(str, perceptron())))
print()

test_input = [1, 1, 0]
test_output = Perceptron.sigmoid(np.dot(np.array(test_input), perceptron.synapses))
print(f"[INPUT]: ({', '.join(list(map(str, test_input)))})")
print(f">>> [RESULT]: {round(test_output[0])} ({test_output[0]})")

"""
[Synapses]:
>>> [9.6734331]
>>> [-0.20781427]
>>> [-4.6298744]

[INPUT]: (1, 1, 0)
>>> [RESULT]: 1 (0.9999225359268941)
"""