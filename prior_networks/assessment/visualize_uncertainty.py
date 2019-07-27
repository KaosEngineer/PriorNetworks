import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)


def plot_histogram(uncertainty_measure, measure_name, ood_uncertainty_measure, save_path=None,
                   log=False, show=True,
                   bins=50, misc=False):
    uncertainty_measure = np.asarray(uncertainty_measure, dtype=np.float128)
    ood_uncertainty_measure = np.asarray(ood_uncertainty_measure, dtype=np.float128)
    # print measure_name
    scores = np.concatenate((uncertainty_measure, ood_uncertainty_measure), axis=0)
    min_score = np.min(scores)
    max_score = np.max(scores)
    plt.hist(uncertainty_measure, bins=bins / 2, range=(min_score, max_score), alpha=0.4)
    plt.hist(ood_uncertainty_measure, bins=bins / 2, range=(min_score, max_score), alpha=0.4)

    if misc == True:
        plt.legend(['Correct', 'Misclassified'])
    else:
        plt.legend(['In-Domain', 'Out-of-Domain'])

    if log == True:
        plt.yscale('log')
    if measure_name == 'max_prob' or measure_name == 'max_prob_log':
        plt.xlim(0.0, 1.0)
        plt.xlabel('Posterior Probability of Mode Class')
    elif measure_name == 'diffential_entropy':
        pass
    else:
        plt.xlim(0.0, -np.log(0.1))

    save_path = os.path.join(save_path, 'Histogram_' + measure_name + '.png')
    plt.savefig(save_path)
    plt.close()
