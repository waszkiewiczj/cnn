import matplotlib.pyplot as plt


def create_accuracy_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy %')
    ax.errorbar(
        x=results['epoch'],
        y=results['mean_accuracy'] * 100,
        yerr=results['var_accuracy']
    )
    return fig


def create_loss_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    ax.errorbar(
        x=results['epoch'],
        y=results['mean_loss'],
        yerr=results['var_loss']
    )
    return fig
