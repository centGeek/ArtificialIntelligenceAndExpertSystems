import pickle
from NeuronNetwork import Network
def save(neuron : Network, filename):
        with open(filename, "wb") as f:
            pickle.dump(neuron, f)
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)