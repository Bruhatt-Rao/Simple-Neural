from brain import network
import numpy as np

n = network.Network([2, 3, 1], learning_rate=0.1)

n.gen()

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
target = np.array([[0], [1], [1], [0]])

n.train(data, target, epochs=10000)

for d in data:
    out = n.feed_forward(d)
    print(f"Input: {d}, Output: {out}")
