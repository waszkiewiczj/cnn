import torch
import torch.optim as optim
import helpers
import trainer.observers as observers


def train_network(network, config, observer=observers.EmptyObserver()):
    device = helpers.get_device()
    network.to(device)
    optimizer = optim.SGD(network.parameters(), lr=config.lr, momentum=config.momentum)
    min_loss = float('inf')
    for epoch in range(config.epochs):
        for iteration, data in enumerate(config.train_set_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = config.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            observer.update(network, epoch, iteration, loss.item())
        validation_loss, validation_accuracy = get_validation_stats(
            network=network,
            validation_set_loader=config.validation_set_loader,
            criterion=config.criterion
        )
        observer.validation_update(network, epoch, validation_loss, validation_accuracy)
        min_loss = min(validation_loss, min_loss)
        if validation_loss / min_loss > 1.2:
            return


def get_validation_stats(network, validation_set_loader, criterion):
    device = helpers.get_device()
    network.to(device)
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in validation_set_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = network(inputs)
            _, predicted = torch.max(outputs, 1)
            loss_val = criterion(outputs, labels)
            loss += loss_val.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return loss, correct / total
