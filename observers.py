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
