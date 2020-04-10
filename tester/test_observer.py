from trainer.observers import ConstFreqObserver
from trainer import get_validation_stats
from tester.test import get_accuracy
import pandas as pd
import torch.utils.data


class TestObserver(ConstFreqObserver):
    def __init__(self, freq, validation_set, criterion):
        super(ConstFreqObserver, self).__init__(freq)
        self.validation_set_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=100,
            shuffle=False
        )
        self.labels = validation_set.targer
        self.criterion = criterion
        self.data = pd.DataFrame({
            'epoch': [],
            'accuracy': [],
            'loss': []
        })

    def freq_update(self, network, epoch, iteration, loss):
        validation_loss, predicted = get_validation_stats(network, self.validation_set_loader, self.criterion)
        validation_acc = get_accuracy(predicted, self.labels)
        self.data.append({
            'epoch': [epoch],
            'accuracy': [validation_acc],
            'loss': [validation_loss]
        })

    def get_results(self):
        grouped = self.data.groupby('epoch')
        means = grouped.mean().add_prefix('mean_')
        vars = grouped.var().add_prefix('var_')
        return means.join(vars).reset_index()
