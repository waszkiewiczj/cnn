import numpy as np
import pandas as pd
import trainer
import helpers
from tester.test_observer import TestObserver
import os
import copy
import tester.plots
import torch


def get_accuracy(predicted, targets):
    correct = np.sum(targets == predicted)
    total = len(predicted)
    return correct / total * 100


def perform_test(config, save_to='test_results'):
    results_dir = f"{save_to}/{config.test_name}"
    os.makedirs(results_dir, exist_ok=True)
    train_config = config.to_train_config()
    all_results = []
    min_loss = float('inf')
    best_network = None
    for i, seed in enumerate(config.seeds):
        test_results, test_network = perform_single_test(
            network=copy.deepcopy(config.network),
            train_config=train_config,
            seed=seed,
            data_collect_freq=config.data_collect_freq
        )
        all_results += [test_results]
        test_results.to_csv(f"{results_dir}/test{str(i).zfill(2)}_results.csv", index=False)
        test_loss = test_results['loss'].min()
        if test_loss < min_loss:
            min_loss = test_loss
            best_network = test_network
            torch.save(best_network.state_dict(), f"{results_dir}/test{str(i).zfill(2)}_model.pth")
        print(f'{i + 1}/{len(config.seeds)} test iterations completed')
    full_results = pd.concat(all_results)
    full_results.to_csv(f"{results_dir}/results.csv", index=False)
    grouped_results = group_results(full_results)
    grouped_results.to_csv(f"{results_dir}/grouped_results.csv", index=False)
    tester.plots.create_accuracy_plot(grouped_results).savefig(f"{results_dir}/accuracy.png")
    tester.plots.create_loss_plot(grouped_results).savefig(f"{results_dir}/loss.png")
    return best_network


def perform_single_test(network, train_config, seed, data_collect_freq):
    observer = TestObserver(freq=data_collect_freq)
    helpers.set_seed(seed=seed)
    best_network = trainer.train_network(
        network=network,
        config=train_config,
        observer=observer
    )
    return observer.get_results(), best_network


def group_results(results):
    grouped = results.groupby('epoch')
    means = grouped.mean().add_prefix('mean_')
    stds = grouped.std().add_prefix('std_')
    return means.join(stds).reset_index()