
# Simple Neural Network

This library implements a simple neural network using Python and NumPy that can be easily modified for different use cases. The network consists of an input layer, a hidden layer, and an output layer. The neural network is trained using the backpropagation algorithm.

## Prerequisites

- Python 3.x
- NumPy

## Installation

Clone the repository:

```sh
git clone https://github.com/Bhim-Rao/Simple-Neural.git
cd simple-neural
```


Install the required packages:

```sh
pip install numpy
```

## Quickstart

To run an example that trains the XOR problem run these commands:

```sh
python main.py
```
This example should output: 
```yaml
Input: [0 0], Output: [0.4984385781118704]
Input: [0 1], Output: [0.5084337861718952]
Input: [1 0], Output: [0.4888554952924806]
Input: [1 1], Output: [0.4988496643606066]
```
## Usage

```py
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
```

This is the example usage for the library

## License

This project is licensed under the MIT License - see the LICENSE file for more details.
