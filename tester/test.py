import numpy as np


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100