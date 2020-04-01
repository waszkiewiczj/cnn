import torch
import torch.utils.data
import numpy as np


def predict(network, testset):
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)
    inputs, *others = next(iter(testset_loader))
    outputs = network(inputs)
    _, predicted = torch.max(outputs, 1)
    return predicted.numpy()


def get_accuracy(net, testset):
    labels = testset.targets
    predicted = predict(net, testset)
    correct = np.sum(labels == predicted)
    total = len(testset)
    return correct / total * 100
