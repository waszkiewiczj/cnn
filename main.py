from torch import nn

import observers
import train
import test
import kaggle
from datasets import load_CIFAR10
from networks import TestNet


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

    train_config = train.TrainConfig(trainset=trainset, batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum,
                               criterion=criterion, seed=seed)

    train.train_network(network=net, config=train_config, observer=observer)

    predicted = test.predict(net, testset)
    accuracy = test.get_accuracy(predicted, testset.targets)
    print('accuracy: %.2f' % accuracy)
    submission_df = kaggle.create_submission_df(testset, predicted)
    submission_df.to_csv('kaggle_submissions/first-try-sub.csv', index=False)


if __name__ == '__main__':
    main()
