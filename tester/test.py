import numpy as np
import trainer
from tester.test_observer import TestObserver
import matplotlib.pyplot as plt


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100


def perform_single_test(config):
    train_config = config.to_train_config()
    observer = TestObserver(config.validation_set, config.data_collect_freq, config.criterion)
    trainer.train_network(config.nerwork, train_config, observer)
    return observer.get_results()
