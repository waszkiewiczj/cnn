from trainer.observers import ConstFreqObserver
from trainer.predict import predict
from tester.test import get_accuracy
import pandas as pd


class TestObserver(ConstFreqObserver):
    def __init__(self, freq, validation_set):
        super(ConstFreqObserver, self).__init__(freq)
        self.validation_set = validation_set
        self.data = pd.DataFrame({
            'epoch': [],
            'accuracy': [],
            'loss': []
        })

    def freq_update(self, network, epoch, iteration, loss):
        predicted = predict(network, self.validation_set)
        acc = get_accuracy(predicted, self.validation_set.target)
        self.data.append({
            'epoch': [epoch],
            'accuracy': [acc],
            'loss': [loss]
        })

    def get_results(self):
        grouped = self.data.groupby('epoch')
        means = grouped.mean().add_prefix('mean_')
        vars = grouped.var().add_prefix('var_')
        return means.join(vars).reset_index()
