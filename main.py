from torch import nn
import trainer
import tester
import datasets.cifar10
import networks
import kaggle


def main():
    trainset, testset = datasets.cifar10.from_kaggle()
    batch_size = 128
    epochs = 2
    lr = 0.001
    momentum = 0.9
    criterion = nn.CrossEntropyLoss()
    seed = 1000
    observer = trainer.observers.DummyPrintObserver()
    net = networks.TestGpuNet()

    train_config = trainer.TrainConfig(trainset=trainset, batch_size=batch_size, epochs=epochs, lr=lr, momentum=momentum,
                                     criterion=criterion, seed=seed)

    trainer.train_network(network=net, config=train_config, observer=observer)

    predicted = tester.predict(network=net, test_set=testset)
    idx_to_class = kaggle.get_idx_to_class_dict(trainset)
    df = kaggle.create_submission_df(predicted=predicted, idx_to_class=idx_to_class)
    df.to_csv('./kaggle_submissions/first-try-sub.csv', index=False)


if __name__ == '__main__':
    main()
