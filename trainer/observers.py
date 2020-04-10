class TrainingObserver:
    def update(self, network, epoch, iteration):
        pass


class EmptyObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration):
        pass


class DummyPrintObserver(TrainingObserver):
    def __init__(self):
        pass

    def update(self, network, epoch, iteration):
        if not iteration % 50:
            print(f'epoch - {epoch}, iteration - {iteration}')


class ConstFreqObserver(TrainingObserver):
    def __init__(self, freq):
        self.freq = freq

    def update(self, network, epoch, iteration):
        if epoch % self.freq == 0:
            self.freq_update(network, epoch, iteration)

    def freq_update(self, network, epoch, iteration):
        pass
