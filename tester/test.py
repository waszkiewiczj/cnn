import numpy as np
import trainer
import helpers
from tester.test_observer import TestObserver
import os


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100


def perform_single_test(config):
    observer = TestObserver(freq=config.data_collect_freq)
    train_config = config.to_train_config()
    for seed in config.seeds:
        helpers.set_seed(seed)
        trainer.train_network(config.network, train_config, observer)
    return observer.get_results()


def save_test_results(test_name, results):
    test_dir_path = f'./test_results/{test_name}/'
    os.makedirs(test_dir_path, exist_ok=True)
    results.to_csv(f'{test_dir_path}results.csv', index=False)
