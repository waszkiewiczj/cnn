import torch.nn as nn
import numpy as np
import json
import networks


class TestConfig:

    @staticmethod
    def __get_supported_networks():
        return {
            'TestNet',
            'TestGpuNet'
        }

    @staticmethod
    def __get_supported_critetions():
        return {
            'CrossEntropyLoss'
        }

    def __init__(
            self,
            test_name,
            network_name,
            split_frac,
            max_epochs,
            learning_rate,
            momentum,
            criterion_name,
            test_count,
            seed,
            data_collect_freq
    ):
        assert network_name in self.__get_supported_networks(), 'Network not supported'
        assert 0 < split_frac <= 1, 'Split frac not in (0, 1]'
        assert 0 < max_epochs, 'Max epochs must be positive'
        assert criterion_name in self.__get_supported_critetions(), 'Criterion not supported'
        assert 0 < test_count, 'Test number must be positive'
        assert 0 < data_collect_freq, 'Data collection frequency must be positive'
        self.test_name = test_name
        self.network = networks.__dict__[network_name]()
        self.split_frac = split_frac
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.criterion = nn.__dict__[criterion_name]()
        self.test_count = test_count
        np.random.seed(seed)
        self.seeds = np.random.randn(test_count)
        self.data_collect_freq = data_collect_freq

    @staticmethod
    def from_json(json_string):
        params = json.loads(json_string)
        return TestConfig(**params)
