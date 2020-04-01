import torch
import torch.utils.data
import numpy as np


def predict(network, testset):
    result_predicted = []
    testset_loader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False)
    with torch.no_grad():
        for data in testset_loader:
            inputs, *others = data
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            result_predicted += predicted.tolist()
    return np.array(result_predicted)


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100
