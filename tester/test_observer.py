from trainer.observers import TrainingObserver
import pandas as pd


class TestObserver(TrainingObserver):
    def __init__(self, freq):
        self.freq = freq
        self.data = {
            'epoch': [],
            'accuracy': [],
            'loss': []
        }

    def update(self, network, epoch, iteration, loss):
        if iteration % 100 == 0:
            print(f'Epoch {epoch}, iteration {iteration}: loss - {loss}')

    def validation_update(self, network, epoch, validation_loss, validation_accuracy):
        if epoch % self.freq == 0:
            self.data['epoch'] += [epoch]
            self.data['accuracy'] += [validation_accuracy]
            self.data['loss'] += [validation_loss]
            print(F'Epoch {epoch} validation: accuracy - {validation_accuracy}, loss -{validation_loss}')

    def get_results(self):
        df = pd.DataFrame(self.data)
        grouped = df.groupby('epoch')
        means = grouped.mean().add_prefix('mean_')
        stds = grouped.std().add_prefix('std_')
        return means.join(stds).reset_index()
