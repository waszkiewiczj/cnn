import torch.utils.data


def split(data, frac):
    train_len = int(len(data) * frac)
    validate_len = len(data) - train_len
    return torch.utils.data.random_split(data, [train_len, validate_len])
