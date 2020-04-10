import torch.optim as optim
import helpers
import trainer.observers as observers


def train_network(network, config, observer=observers.EmptyObserver()):
    device = helpers.get_device()
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
            observer.update(network, epoch, iteration)
