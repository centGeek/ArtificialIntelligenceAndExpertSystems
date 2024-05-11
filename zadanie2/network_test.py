import numpy as np
from functions import sigmoid
class Network_Test():
    def test(self, test_data):
        test_results = [(np.argmax(self.forward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def forward(self, result):
        for i in range(len(self.biases)):
                bias = self.biases[i]
                weight = self.weights[i]
                a = sigmoid(np.dot(weight, a) + bias)
        return result