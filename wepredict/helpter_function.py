import pickle

def save_pickle(object, path):
    """Save obejct."""
    pickle.dump(object, open(path, 'wb'))

def load_pickle(path):
    """Save obejct."""
    return pickle.load(open(path, 'rb'))
