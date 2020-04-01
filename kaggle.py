import numpy as np
import pandas as pd


def get_idx_to_class_dict(dataset):
    return {
        dataset.class_to_idx[label]: label
        for label in dataset.class_to_idx
    }


def map_to_labels(idx_to_class, output):
    return np.array([ idx_to_class[value] for value in output])


def create_submission_df(dataset, predicted):
    idx_to_class = get_idx_to_class_dict(dataset)
    labeled = map_to_labels(idx_to_class, predicted)
    return pd.DataFrame({
        'id': range(1, len(labeled) + 1),
        'label': labeled
    })
