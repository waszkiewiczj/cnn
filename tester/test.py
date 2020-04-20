import numpy as np
import trainer
import helpers
from tester.test_observer import TestObserver
import os
import tester.plots
import copy


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100


def perform_single_test(config):
    observer = TestObserver(freq=config.data_collect_freq)
    train_config = config.to_train_config()
    for i, seed in enumerate(config.seeds):
        helpers.set_seed(seed)
        network = copy.deepcopy(config.network)
        trainer.train_network(network, train_config, observer)
        print(f'{i + 1}/{len(config.seeds)} test completed')
    return observer.get_results(), observer.get_raw_results()
