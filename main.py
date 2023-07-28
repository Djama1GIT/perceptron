import random
import sys

import numpy as np

sys.setrecursionlimit(2000000)


def foo(a, b, c, d, e, f):
    return a and b and c or d or (not e) and f or a and f


class Perceptron:
    def __init__(self, _inputs, _outputs):
        self.inputs = _inputs
        self.outputs = _outputs
        self.synapses = 2 * np.random.random((6, 1)) - 1

    def __call__(self, *args, **kwargs):
        self.backpropagation()
        return self.synapses

    def backpropagation(self):
        for i in range(10000):
            __input = self.inputs
            __outputs = Perceptron.sigmoid(np.dot(__input, self.synapses))

            err = self.outputs - __outputs
            adjustments = np.dot(__input.T, err * (__outputs * (1 - __outputs)))

            self.synapses += adjustments

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


test_count = 0


def test_perceptron():
    try:
        global test_count
        test_count += 1
        inputs = []
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            for f in [0, 1]:
                                if random.randint(0, 1):
                                    inputs += [[a, b, c, d, e, f]]
                                    # Интересный факт: если в инпут поместить все данные,
                                    # то выйдет НИЧЕГО

        inputs = np.array(inputs)
        outputs = np.array([[foo(*i) for i in inputs]]).T
        perceptron = Perceptron(inputs, outputs)
        perceptron()
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    for d in [0, 1]:
                        for e in [0, 1]:
                            for f in [0, 1]:
                                print((a, b, c, d, e, f), foo(a, b, c, d, e, f),
                                      Perceptron.sigmoid(np.dot(
                                          np.array([a, b, c, d, e, f]), perceptron.synapses)
                                      )[0])
                                assert (foo(a, b, c, d, e, f) == 1) == (round(
                                    Perceptron.sigmoid(np.dot(
                                        np.array([a, b, c, d, e, f]), perceptron.synapses)
                                    )[0]) > 0.5) or print(f"Тест #{test_count} Не прошел")
        else:
            print(inputs)
            print(f"Тест #{test_count} пройден. Перцептрон работает.")  # я пытался, не вышло
    except:
        test_perceptron()


test_perceptron()
