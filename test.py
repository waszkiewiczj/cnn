import torch
import torch.utils.data
import numpy as np


def predict(network, testset):
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    inputs, *others = next(iter(testset_loader))
    outputs = network(inputs)
    _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100
