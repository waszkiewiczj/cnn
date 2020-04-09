class TrainingObserver:
    def update(self, network, epoch, iteration, loss):
        pass


class EmptyObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration, loss):
        pass


class DummyPrintObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration, loss):
        if not iteration % 50:
            print(f'epoch - {epoch}, iteration - {iteration}, loss - {loss}')


class ConstFreqObserver(TrainingObserver):
    def __init__(self, freq):
        self.freq = freq

    def update(self, network, epoch, iteration, loss):
        if epoch % self.freq == 0:
            self.freq_update(network, epoch, iteration, loss)

    def freq_update(self, network, epoch, iteration, loss):
        pass
