import torch
import torch.optim as optim
import helpers
import trainer.observers as observers


def train_network(network, config, observer=observers.EmptyObserver()):
    device = helpers.get_device()
    network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=config.lr, momentum=config.momentum)
    last_lost = float('inf')
    for epoch in range(config.epochs):
        for iteration, data in enumerate(config.train_set_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = config.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            observer.update(network, epoch, iteration, loss.item())
        validation_loss, _ = get_validation_stats(network, config.validation_set_loader, config.criterion)
        if validation_loss > last_lost:
            return
        last_lost = validation_loss


def get_validation_stats(network, validation_set_loader, criterion):
    device = helpers.get_device()
    network.to(device)
    result_predicted = []
    loss = 0
    with torch.no_grad():
        for data in validation_set_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            loss_val = criterion(outputs, labels)
            loss += loss_val.item()
            result_predicted += predicted.tolist()
    return loss, predicted
