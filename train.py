import torch
import torch.utils.data
import torch.backends.cudnn
import torch.cuda
import torch.optim as optim
import numpy as np
import random
import observers


class TrainConfig:
    def __init__(self, trainset, batch_size, epochs, lr, momentum, criterion, seed=None):
        self.__set_seed(seed)
        self.trainset_loader = torch.utils.data.DataLoader(
            trainset,
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


def train_network(network, config, observer=observers.EmptyObserver()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=config.lr, momentum=config.momentum)
    for epoch in range(config.epochs):
        for iteration, data in enumerate(config.trainset_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = config.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            observer.update(network, epoch, iteration, loss.item())


def get_accuracy(network, testset):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(device)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total * 100
