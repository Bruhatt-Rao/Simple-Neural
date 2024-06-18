import numpy as np
from .neuron import Neuron
from .utils import Config
import os
name = os.name
def clear():
    if name == "nt":
        os.system("cls")
    else:
        os.system('clear')

class Network:
    def __init__(self, size, learning_rate=0.01):
        self.size = size
        self.learning_rate = learning_rate
        self.middle = []
        self.output = []
        self.config = Config()
        self.count = 0

    def gen(self):
        self.middle = [Neuron(self.size[0]) for _ in range(self.size[1])]
        self.output = [Neuron(self.size[1]) for _ in range(self.size[2])]

    def feed_forward(self, inp):
        if len(inp) != self.size[0]:
            raise ValueError("Input does not match input size")

        middle_outputs = [neuron.output(inp) for neuron in self.middle]
        final_outputs = [neuron.output(np.array(middle_outputs), activation=self.config.out_activation) for neuron in self.output]

        return final_outputs

    def backpropagate(self, inputs, targets):
        for i, neuron in enumerate(self.output):
            neuron.calculate_delta(target=targets[i], activation=self.config.out_activation)

        for i, neuron in enumerate(self.middle):
            downstream_weights = np.array([n.w[i] for n in self.output])
            downstream_deltas = np.array([n.delta for n in self.output])
            neuron.calculate_delta(downstream_weights=downstream_weights, downstream_deltas=downstream_deltas)

        for neuron in self.output:
            neuron.update_weights(np.array([n.output_cache for n in self.middle]), self.learning_rate)

        for neuron in self.middle:
            neuron.update_weights(np.array(inputs), self.learning_rate)

    def train(self, data, targets, epochs=10000):
        for _ in range(epochs):
            clear()
            print(f"{int((self.count/epochs) * 100)}% : {self.count}/{epochs}")
            self.count += 1
            for inp, target in zip(data, targets):
                self.feed_forward(inp)
                self.backpropagate(inp, target)
