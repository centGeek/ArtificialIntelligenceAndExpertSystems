import pickle
from NeuronNetwork import NeuronNetwork
def save(neuron : NeuronNetwork, filename):
        with open(filename, "wb") as f:
            pickle.dump(neuron, f)
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)