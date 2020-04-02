from torch import nn
import observers
import train
import test
import datasets.cifar10
import networks
import kaggle


def main():
    trainset, testset = datasets.cifar10.from_kaggle()
    batch_size = 4
    epochs = 2
    lr = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    seed = 1000
    observer = observers.DummyPrintObserver()
    net = networks.TestNet()


    train_config = train.TrainConfig(trainset=trainset, batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum,
                               criterion=criterion, seed=seed)

    train.train_network(network=net, config=train_config, observer=observer)

    predicted = test.predict(network=net, testset=testset)
    idx_to_class = kaggle.get_idx_to_class_dict(trainset)
    df = kaggle.create_submission_df(predicted=predicted, idx_to_class=idx_to_class)
    df.to_csv('./kaggle_submissions/first-try-sub.csv', index=False)


if __name__ == '__main__':
    main()
