import torch
import torch.utils.data
import numpy as np
import pandas as pd
import helpers


def predict(network, set_loader):
    device = helpers.get_device()
    network.to(device)
    result_predicted = []
    with torch.no_grad():
        for data in set_loader:
            inputs = data[0].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            result_predicted += predicted.tolist()
            print(f"{len(result_predicted)} records evaluated")
    return np.array(result_predicted)


def get_idx_to_class_dict(dataset):
    return {
        dataset.class_to_idx[label]: label
        for label in dataset.class_to_idx
    }


def map_to_labels(idx_to_class, output):
    return np.array([idx_to_class[value] for value in output])


def create_submission_df(predicted, idx_to_class):
    labeled = map_to_labels(idx_to_class, predicted)
    return pd.DataFrame({
        'id': range(1, len(labeled) + 1),
        'label': labeled
    })
