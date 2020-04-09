import torch
import torch.utils.data
import numpy as np
import helpers


def predict(network, test_set):
    device = helpers.get_device()
    network.to(device)
    result_predicted = []
    test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)
    with torch.no_grad():
        for data in test_set_loader:
            inputs = data[0].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            result_predicted += predicted.tolist()
    return np.array(result_predicted)