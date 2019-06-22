import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata
from matplotlib import cm



def plot_training_progress(train_loss, train_accuracy, test_loss, test_accuracy, test_steps, title=''):
    """
    Plot the loss and test loss and accuracy. Assumes train loss sampled at every step.
    """
    f, axarr = plt.subplots(ncols=2, figsize=(14, 6))
    f.suptitle(title, fontsize=16)

    colors = sns.cubehelix_palette(2, start=-0.4)
    axarr[0].plot(np.arange(len(train_loss)), train_loss, color=colors[0], label="Train Loss", alpha=0.7)
    axarr[0].plot(test_steps, test_loss, color=colors[1], label="Test Loss", alpha=0.8)
    axarr[0].legend(loc='upper right')

    axarr[1].plot(np.arange(len(train_loss)), train_accuracy, color=colors[0], label="Train Accuracy", alpha=0.7)
    axarr[1].plot(test_steps, test_accuracy, color=colors[1], label="Test Accuracy", alpha=0.8)
    axarr[1].legend(loc='lower right')
    axarr[1].set_ylim(0., 1.0)
    plt.show()
    return axarr


def visualise_uncertainty(uncertainty_vals, xrange=(-500, 500), yrange=(-500, 500), ax=None, title='', cmap=None,
                          dataset_to_overlay=False):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    ax.set(aspect='equal')
    cs = ax.matshow(uncertainty_vals, extent=[*xrange, *yrange], cmap=cmap,
                   interpolation='bilinear', origin='lower')

    if dataset_to_overlay:
        points = dataset_to_overlay.x
        ax.scatter(points[:, 0], points[:, 1], color=(1., 1., 1.), alpha=0.5, s=10)
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)

    ax.figure.colorbar(cs, ax=ax, aspect=40)
    ax.set_title(title, fontsize=16)
    ax.grid(False)
    return ax


def plot_contourf(inputs, uncertainty, ext, res, show=True, name='uncertainty'):
    xi = np.linspace(-ext, ext, res)
    yi = np.linspace(-ext, ext, res)

    zi_entropy = griddata(inputs, uncertainty, (xi[None, :], yi[:, None]), method='linear')
    plt.contourf(xi, yi, zi_entropy, cmap=cm.Blues, alpha=0.9)
    plt.xlim(-ext, ext)
    plt.ylim(-ext, ext)
    #plt.title(name)
    plt.colorbar()
    if show: plt.show()
    else: plt.savefig(name+'.png', bbox_inches='tight')
    plt.close()
