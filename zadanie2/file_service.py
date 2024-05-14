import pickle
from NeuronNetwork import Network
def save(network : Network, filename):
        with open(filename, "wb") as f:
            pickle.dump(network, f)
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)