import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)


# TODO DECIDE HOW TO COMBINE THIS WITH ADV STUFF
def eval_ood_detect(domain_labels, in_uncertainties, out_uncertainties, save_path,
                    classes_flipped=None, adversarial=False):
    # if adversarial == True:
    #     functions = [plot_mod_roc_curve]
    for mode in ['PR', 'ROC']:
        for key in in_uncertainties.keys():
            pos_label = 1
            if key == 'confidence':
                pos_label = 0

                # try:
                # if adversarial == True:
                #     function(domain_labels, in_uncertainties[key][0], out_uncertainties[key][0], classes_flipped,
                #              in_uncertainties[key][1],
                #              save_path=save_path, pos_label=pos_label, show=show)
            ood_detect(domain_labels, in_uncertainties[key], out_uncertainties[key], key, mode=mode,
                       save_path=save_path, pos_label=pos_label)
            # except:
            #     pass


def ood_detect(domain_labels, in_measure, out_measure, measure_name, save_path, mode, pos_label=1):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUPR using ' + measure_name + ": " + str(np.round(aupr * 100.0, 1)) + '\n')
        np.savetxt(os.path.join(save_path, measure_name + '_recall.txt'), recall)
        np.savetxt(os.path.join(save_path, measure_name + '_precision.txt'), precision)

        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'PR_' + measure_name + '.png'))
        plt.close()

    elif mode == 'ROC':
        fpr, tpr, thresholds = roc_curve(domain_labels, scores)
        roc_auc = roc_auc_score(domain_labels, scores)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUROC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')
        np.savetxt(os.path.join(save_path, measure_name + '_trp.txt'), tpr)
        np.savetxt(os.path.join(save_path, measure_name + '_frp.txt'), fpr)

        plt.plot(fpr, tpr)
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'ROC_' + measure_name + '.png'))
        plt.close()


# TODO: Fix adversarial detection stuff later...
def mod_roc_curve(y_true, y_score, class_flipped, pos_label=1):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
        class_flipped : array, shape = [n_samples]
        whether the predicted class was flipped or not
    pos_label : int or str, default=None
        The label of the positive class
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """

    # y_true = column_or_1d(y_true)
    # y_score = column_or_1d(y_score)

    # make y_true a boolean vector
    # y_true = (y_true == pos_label)
    # print y_true

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    class_flipped = class_flipped[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    tpr = tps / np.float(tps[-1])

    y_true = 1 - y_true
    total_neg = np.float(np.sum(y_true))
    y_true = y_true * class_flipped
    # print total_neg, np.sum(y_true)
    fps = np.cumsum(y_true)[threshold_idxs]
    fpr = fps / total_neg

    fpr = np.r_[fpr, 1.0]
    tpr = np.r_[tpr, 1.0]

    auc_score = auc(fpr, tpr, reorder=True)

    return auc_score, fpr, tpr, y_score[threshold_idxs]


def plot_mod_roc_curve(domain_labels, in_measure, out_measure, class_flipped, measure_name,
                       save_path, pos_label=1, show=True):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float128)
    not_flipped = np.asarray(np.zeros_like(in_measure), dtype=np.int32)
    class_flipped = np.r_[not_flipped, class_flipped]
    if pos_label != 1:
        scores = -1.0 * scores

    roc_auc, fpr, tpr, thresholds = mod_roc_curve(domain_labels, scores,
                                                  class_flipped=class_flipped)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write(
            'MOD ROC AUC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')
    np.savetxt(os.path.join(save_path, measure_name + '_trp.txt'), tpr)
    np.savetxt(os.path.join(save_path, measure_name + '_frp.txt'), fpr)

    plt.plot(fpr, tpr)
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    if show:
        save_path = os.path.join(save_path, 'ROC_' + measure_name + '.png')
        plt.savefig(save_path)
    plt.close()
