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

# TODO: Maybe find a better name??
def eval_misc_detect(class_labels, class_probs, uncertainties, save_path, misc_positive=True):
    for mode in ['PR', 'ROC']:
        for key in uncertainties.keys():
            pos_label = 1
            if key == 'confidence':
                pos_label = 0

            try:
                misc_detect(class_labels, class_probs, uncertainties[key], key,
                            save_path=save_path, mode=mode, pos_label=pos_label, misc_positive=misc_positive)
            except:
                pass

def misc_detect(class_labels, class_probs, measure, measure_name, save_path, mode, pos_label=1, misc_positive=None):
    # TODO: Is this necessary???
    measure = np.asarray(measure, dtype=np.float128)[:, np.newaxis]
    min_measure = np.min(measure)
    if min_measure < 0.0: measure += abs(min_measure)
    measure = np.log(measure + 1e-8)

    class_probs = np.asarray(class_probs, dtype=np.float64)
    class_preds = np.argmax(class_probs, axis=1)[:, np.newaxis]
    class_labels = class_labels[:, np.newaxis]

    if misc_positive == True:
        rightwrong = np.asarray(class_labels != class_preds, dtype=np.int32)
    else:
        rightwrong = np.asarray(class_labels == class_preds, dtype=np.int32)

    if pos_label != 1: measure *= -1.0

    if mode=='PR':
        precision, recall, thresholds = precision_recall_curve(rightwrong, measure)
        aupr = auc(recall, precision)
        np.round(aupr, 4)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUPR using ' + measure_name + ": " + str(np.round(aupr * 100.0, 1)) + '\n')

        np.savetxt(os.path.join(save_path, measure_name + '_recall.txt'), recall)
        np.savetxt(os.path.join(save_path, measure_name + '_precision.txt'), precision)

        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'PR_curve_' + measure_name + '.png'))
        plt.close()

    elif mode == 'ROC':
        fpr, tpr, thresholds = roc_curve(rightwrong, measure)
        roc_auc = roc_auc_score(rightwrong, measure)
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AUROC using ' + measure_name + ": " + str(np.round(roc_auc * 100.0, 1)) + '\n')

        np.savetxt(os.path.join(save_path, measure_name + '_tpr.txt'), tpr)
        np.savetxt(os.path.join(save_path, measure_name + '_fpr.txt'), fpr)

        plt.plot(fpr, tpr)
        plt.xlabel('False Positive')
        plt.ylabel('True Positive')
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.savefig(os.path.join(save_path, 'ROC_curve_' + measure_name + '.png'))
        plt.close()

    else:
        print('Inappropriate experiment mode')
