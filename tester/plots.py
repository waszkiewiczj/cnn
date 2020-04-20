import matplotlib.pyplot as plt
import numpy as np


def create_accuracy_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0, 100)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy %')
    ax.set_yticks(np.r_[0:100:10])
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(len(results), 16)))
    ax.grid(True, axis='y')
    ax.errorbar(
        x=results['epoch'],
        y=results['mean_accuracy'] * 100,
        yerr=results['std_accuracy']
    )
    return fig


def create_loss_plot(results):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    ax.xaxis.set_major_locator(plt.MaxNLocator(min(len(results), 16)))
    ax.grid(True, axis='y')
    ax.errorbar(
        x=results['epoch'],
        y=results['mean_loss'],
        yerr=results['std_loss']
    )
    return fig
