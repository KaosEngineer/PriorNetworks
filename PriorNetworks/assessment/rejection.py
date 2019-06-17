import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import auc
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)


def reject_MSE(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE
    error = (preds - targets) ** 2
    MSE_0 = np.mean(error)
    # print 'BASE MSE', MSE_0

    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, 0.0]]
    results_var = [[0.0, 0.0]]
    results_min = [[0.0, 0.0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])

    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max], reorder=True)
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min], reorder=True)

    plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
    plt.xlim(0.0, 1.0)
    if show == True:
        plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
    plt.close()
    plt.plot([x[0] for x in results_max], [x[1] for x in results_max], '^',
             [x[0] for x in results_var], [x[1] for x in results_var], 'o',
             [x[0] for x in results_min], [x[1] for x in results_min], '--')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Pearson Correlation')
    if show==True:
        plt.savefig(os.path.join(save_path, "MSE Rejection Curve using " + measure_name), bbox_inches='tight')
    plt.close()

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
    if save_path is not None:
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('MSE ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')
    return AUC_RR


def reject_AE(targets, preds, measure, measure_name, save_path, grd=1.0, pos_label=1, show=True):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE
    error = (targets-preds)**2
    MSE_0 = np.mean(np.greater_equal(np.abs(targets-preds), grd, dtype=np.float32))
    # print 'BASE MSE', MSE_0

    # Create array
    array = np.concatenate((preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)


    results_min = [[float(i)/float(array.shape[0]), MSE_0*(array.shape[0]-float(i))/float(array.shape[0])] for i in range(array.shape[0]+1)]

    results_max = []
    for i in range(array.shape[0]+1):
        if i <= int(MSE_0*array.shape[0]):
            results_max.append([float(i)/float(array.shape[0]), np.maximum((array.shape[0]*MSE_0-float(i))/float(array.shape[0]),0.0)])
        else:
            results_max.append([float(i)/float(array.shape[0]), 0.0])


    results_var = [[0.0, MSE_0]]
    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty
    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean(np.greater_equal(np.abs(x - sorted_array[:, 1]), grd, dtype=np.float32))
        results_var.append([float(i) / float(array.shape[0]), mse])


    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max], reorder=True)
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min], reorder=True)

    # plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
    # plt.xlim(0.0, 1.0)
    # if show == True:
    #     plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
    # plt.close()
    plt.plot([x[0] for x in results_max], [x[1] for x in results_max], '^',
             [x[0] for x in results_var], [x[1] for x in results_var], 'o',
             [x[0] for x in results_min], [x[1] for x in results_min], 'k--')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'])
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, MSE_0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('% > 1 GRD AE')
    if show==True:
        plt.savefig(os.path.join(save_path, "AE Rejection Curve using " + measure_name+'.png'), bbox_inches='tight')
    plt.close()

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
    if save_path is not None:
        with open(os.path.join(save_path, 'results.txt'), 'a') as f:
            f.write('AE ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')
    return AUC_RR

def reject_PCC(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE and PCC
    error = (preds - targets) ** 2
    P_0 = pearsonr(preds, targets)[0]
    # print 'BASE MSE', P_0
    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, P_0]]
    results_var = [[0.0, P_0]]
    results_min = [[0.0, P_0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        p = pearsonr(x, sorted_array[:, 1])[0]
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), p])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), P_0 + (1.0 - P_0) * float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        p = pearsonr(x, sorted_array[:, 1])[0]
        results_var.append([float(i) / float(array.shape[0]), p])

    # print 'Rho', rho(optimal_ranking, uncertainty_ranking)
    # print 'tau', tau(optimal_ranking, uncertainty_ranking)
    max_auc = auc([x[0] for x in results_max], [x[1] - P_0 for x in results_max], reorder=True)
    var_auc = auc([x[0] for x in results_var], [x[1] - P_0 for x in results_var], reorder=True)
    min_auc = auc([x[0] for x in results_min], [x[1] - P_0 for x in results_min], reorder=True)

    plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
    plt.xlim(0.0, 1.0)
    if show:
        plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
    plt.close()
    plt.plot([x[0] for x in results_max], [x[1] for x in results_max], 'b^',
             [x[0] for x in results_var], [x[1] for x in results_var], 'ro',
             [x[0] for x in results_min], [x[1] for x in results_min], 'go')
    plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4)
    plt.xlim(0.0, 1.0)
    plt.ylim(P_0, 1.0)
    plt.xlabel('Rejection Fraction')
    plt.ylabel('Pearson Correlation')
    if show:
        plt.savefig(os.path.join(save_path, "PCC Rejection Curve using " + measure_name), bbox_inches='tight')
    plt.close()

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('PCC ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')
