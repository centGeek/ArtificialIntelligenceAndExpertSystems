import pickle
def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)