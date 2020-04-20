import numpy as np
import trainer
import helpers
from tester.test_observer import TestObserver
from tester.test_results_saver import TestResultsSaver
import copy


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100


def perform_single_test(config):
    observer = TestObserver(freq=config.data_collect_freq)
    train_config = config.to_train_config()
    saver = TestResultsSaver('test_results')
    for i, seed in enumerate(config.seeds):
        helpers.set_seed(seed)
        network = copy.deepcopy(config.network)
        trainer.train_network(network, train_config, observer)
        print(f'{i + 1}/{len(config.seeds)} test completed')
        saver.save_partial_results(i, observer.get_results())
    results = observer.get_results()
    saver.save_full_results(results)
    return results
