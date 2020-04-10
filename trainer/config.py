import torch
import torch.utils.data
import torch.backends.cudnn
import numpy as np
import random


class TrainConfig:
    def __init__(self, train_set, validation_set, batch_size, epochs, lr, momentum, criterion, seed=None):
        self.__set_seed(seed)
        self.train_set_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        self.validation_set_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=True
        )
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.criterion = criterion

    def __set_seed(self, seed):
        """
        Set seed for all torch func and methods.
        See ref: https://github.com/pytorch/pytorch/issues/7068
        """
        torch.manual_seed(seed)
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = seed is None
        torch.backends.cudnn.deterministic = seed is not None