import numpy as np

class Neuron:
    def __init__(self, inputs):
        self.w = np.random.rand(inputs)
        self.b = np.random.rand()
        self.output_cache = 0
        self.delta = 0
        self.count = 0

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sig(self, n):
        return 1 / (1 + np.exp(-n))

    def sig_derivative(self, n):
        return n * (1 - n)

    def RELU(self, inp):
        return np.maximum(0, inp)

    def RELU_derivative(self, inp):
        return np.where(inp > 0, 1, 0)

    def output(self, inp, activation="RELU"):
        s = np.dot(inp, self.w) + self.b
        self.output_cache = self.RELU(s) if activation == "RELU" else self.sig(s)
        return self.output_cache

    def calculate_delta(self, target=None, downstream_weights=None, downstream_deltas=None, activation="RELU"):
        if target is not None:
            self.delta = (self.output_cache - target) * (self.RELU_derivative(self.output_cache) if activation == "RELU" else self.sig_derivative(self.output_cache))
        else:
            self.delta = np.dot(downstream_weights, downstream_deltas) * (self.RELU_derivative(self.output_cache) if activation == "RELU" else self.sig_derivative(self.output_cache))

    def update_weights(self, inputs, learning_rate):
        self.w -= learning_rate * self.delta * inputs
        self.b -= learning_rate * self.delta
