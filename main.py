from torch import nn

import observers
from datasets import load_CIFAR10
from networks import TestNet
from train import TrainConfig, train_network


def main():
    trainset, testset = load_CIFAR10()
    batch_size = 4
    epochs = 2
    lr = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    seed = 1000
    observer = observers.DummyPrintObserver()
    net = TestNet()

    train_config = TrainConfig(trainset=trainset, batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum,
                               criterion=criterion, seed=seed)

    train_network(network=net, config=train_config, observer=observer)


if __name__ == '__main__':
    main()
