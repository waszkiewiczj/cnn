import torch.nn as nn
import numpy as np
import json
import networks
import datasets
from trainer import TrainConfig


class TestConfig:

    @staticmethod
    def __get_supported_networks():
        return {
            'testnet',
            'linear'
        }

    @staticmethod
    def __get_supported_transfers():
        return {
            'empty',
            'resnet',
            'alexnet',
            'vgg'
        }

    @staticmethod
    def __get_supported_critetions():
        return {
            'CrossEntropyLoss'
        }

    def __init__(
            self,
            test_name,
            transfer_name,
            network_name,
            transfer_pretrained,
            split_frac,
            max_epochs,
            batch_size,
            learning_rate,
            momentum,
            criterion_name,
            test_count,
            seed,
            data_collect_freq
    ):
        assert network_name in self.__get_supported_networks(), 'Network not supported'
        assert transfer_name in self.__get_supported_transfers(), 'Transfer not supported'
        assert 0 < split_frac <= 1, 'Split frac not in (0, 1]'
        assert 0 < max_epochs, 'Max epochs must be positive'
        assert criterion_name in self.__get_supported_critetions(), 'Criterion not supported'
        assert 0 < test_count, 'Test number must be positive'
        assert 0 < data_collect_freq, 'Data collection frequency must be positive'
        self.test_name = test_name
        self.network, input_size = networks.build(transfer_name,network_name,transfer_pretrained)
        data_set = datasets.cifar10.from_kaggle(train=True, input_size=input_size)
        self.train_set, self.validation_set = datasets.split(data_set, frac=split_frac)
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.criterion = nn.__dict__[criterion_name]()
        self.test_count = test_count
        np.random.seed(seed)
        self.seeds = np.random.randint(1000000, size=test_count)
        self.data_collect_freq = data_collect_freq

    def to_train_config(self):
        return TrainConfig(
            train_set=self.train_set,
            validation_set=self.validation_set,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            lr=self.learning_rate,
            momentum=self.momentum,
            criterion=self.criterion
        )

    @staticmethod
    def from_json(json_string):
        params = json.loads(json_string)
        return TestConfig(**params)
