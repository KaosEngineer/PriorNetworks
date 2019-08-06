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


# TODO: Maybe find a better name??
def eval_rejection_ratio_class(labels, probs, uncertainties, save_path):
    for key in uncertainties.keys():
        rev = False
        if key == 'confidence':
            rev = True

        #try:
        reject_class(labels, probs, uncertainties[key], key, save_path=save_path, rev=rev)
        #except:
        #    pass


def reject_class(labels, probs, measure, measure_name: str, save_path: str, rev: bool, show=True):
    # Get predictions
    preds = np.argmax(probs, axis=1)

    if rev:
        inds = np.argsort(measure)[::-1]
    else:
        inds = np.argsort(measure)

    total_data = np.float(preds.shape[0])
    errors, percentages = [], []

    for i in range(preds.shape[0]):
        errors.append(np.sum(
            np.asarray(labels[inds[:i]] != preds[inds[:i]], dtype=np.float32)) * 100.0 / total_data)
        percentages.append(float(i + 1) / total_data * 100.0)
    errors, percentages = np.asarray(errors)[:, np.newaxis], np.asarray(percentages)

    base_error = errors[-1]
    n_items = errors.shape[0]
    auc_uns = 1.0 - auc(percentages / 100.0, errors[::-1] / 100.0)

    random_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(n_items)) for i in range(n_items)],
        dtype=np.float32)
    auc_rnd = 1.0 - auc(percentages / 100.0, random_rejection / 100.0)
    orc_rejection = np.asarray(
        [base_error * (1.0 - float(i) / float(base_error / 100.0 * n_items)) for i in
         range(int(base_error / 100.0 * n_items))], dtype=np.float32)
    orc = np.zeros_like(errors)
    orc[0:orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1.0 - auc(percentages / 100.0, orc / 100.0)

    if show:
        plt.plot(percentages, orc, lw=2)
        plt.fill_between(percentages, orc, random_rejection, alpha=0.5)
        plt.plot(percentages, errors[::-1], lw=2)
        plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.0)
        plt.plot(percentages, random_rejection, 'k--', lw=2)
        plt.legend(['Oracle', 'Uncertainty', 'Random'])
        plt.xlabel('Percentage of predictions rejected to oracle')
        plt.ylabel('Classification Error (%)')
        plt.savefig('Rejection-Curve-oracle.png', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

        plt.plot(percentages, orc, lw=2)
        plt.fill_between(percentages, orc, random_rejection, alpha=0.0)
        plt.plot(percentages, errors[::-1], lw=2)
        plt.fill_between(percentages, errors[::-1], random_rejection, alpha=0.5)
        plt.plot(percentages, random_rejection, 'k--', lw=2)
        plt.legend(['Oracle', 'Uncertainty', 'Random'])
        plt.xlabel('Percentage of predictions rejected to oracle')
        plt.ylabel('Classification Error (%)')
        plt.savefig('Rejection-Curve-uncertainty.png', bbox_inches='tight', dpi=300)
        # plt.show()
        plt.close()

    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0

    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write(f'Rejection Ratio using {measure_name}: {np.round(rejection_ratio, 1)}\n')

#
# def reject_MSE(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
#     if pos_label != 1:
#         measure_loc = -1.0 * measure
#     else:
#         measure_loc = measure
#     preds = np.squeeze(preds)
#     # Compute total MSE
#     error = (preds - targets) ** 2
#     MSE_0 = np.mean(error)
#     # print 'BASE MSE', MSE_0
#
#     # Create array
#     array = np.concatenate(
#         (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)
#
#     # Results arrays
#     results_max = [[0.0, 0.0]]
#     results_var = [[0.0, 0.0]]
#     results_min = [[0.0, 0.0]]
#
#     optimal_ranking = array[:, 2].argsort()
#     sorted_array = array[optimal_ranking]  # Sort by error
#
#     for i in range(1, array.shape[0]):
#         x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
#         mse = np.mean((x - sorted_array[:, 1]) ** 2)
#         # Best rejection
#         results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
#         # Random Rejection
#         results_min.append([float(i) / float(array.shape[0]), float(i) / float(array.shape[0])])
#
#     uncertainty_ranking = array[:, 3].argsort()
#     sorted_array = array[uncertainty_ranking]  # Sort by uncertainty
#
#     for i in range(1, array.shape[0]):
#         x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
#         mse = np.mean((x - sorted_array[:, 1]) ** 2)
#         results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
#
#     max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max], reorder=True)
#     var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
#     min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min], reorder=True)
#
#     plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
#     plt.xlim(0.0, 1.0)
#     if show == True:
#         plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
#     plt.close()
#     plt.plot([x[0] for x in results_max], [x[1] for x in results_max], '^',
#              [x[0] for x in results_var], [x[1] for x in results_var], 'o',
#              [x[0] for x in results_min], [x[1] for x in results_min], '--')
#     plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4)
#     plt.xlim(0.0, 1.0)
#     plt.ylim(0.0, 1.0)
#     plt.xlabel('Rejection Fraction')
#     plt.ylabel('Pearson Correlation')
#     if show==True:
#         plt.savefig(os.path.join(save_path, "MSE Rejection Curve using " + measure_name), bbox_inches='tight')
#     plt.close()
#
#     AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
#     if save_path is not None:
#         with open(os.path.join(save_path, 'results.txt'), 'a') as f:
#             f.write('MSE ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')
#     return AUC_RR
#
#
# def reject_AE(targets, preds, measure, measure_name, save_path, grd=1.0, pos_label=1, show=True):
#     if pos_label != 1:
#         measure_loc = -1.0 * measure
#     else:
#         measure_loc = measure
#     preds = np.squeeze(preds)
#     # Compute total MSE
#     error = (targets-preds)**2
#     MSE_0 = np.mean(np.greater_equal(np.abs(targets-preds), grd, dtype=np.float32))
#     # print 'BASE MSE', MSE_0
#
#     # Create array
#     array = np.concatenate((preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)
#
#
#     results_min = [[float(i)/float(array.shape[0]), MSE_0*(array.shape[0]-float(i))/float(array.shape[0])] for i in range(array.shape[0]+1)]
#
#     results_max = []
#     for i in range(array.shape[0]+1):
#         if i <= int(MSE_0*array.shape[0]):
#             results_max.append([float(i)/float(array.shape[0]), np.maximum((array.shape[0]*MSE_0-float(i))/float(array.shape[0]),0.0)])
#         else:
#             results_max.append([float(i)/float(array.shape[0]), 0.0])
#
#
#     results_var = [[0.0, MSE_0]]
#     uncertainty_ranking = array[:, 3].argsort()
#     sorted_array = array[uncertainty_ranking]  # Sort by uncertainty
#     for i in range(1, array.shape[0]):
#         x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
#         mse = np.mean(np.greater_equal(np.abs(x - sorted_array[:, 1]), grd, dtype=np.float32))
#         results_var.append([float(i) / float(array.shape[0]), mse])
#
#
#     max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max], reorder=True)
#     var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var], reorder=True)
#     min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min], reorder=True)
#
#     # plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
#     # plt.xlim(0.0, 1.0)
#     # if show == True:
#     #     plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
#     # plt.close()
#     plt.plot([x[0] for x in results_max], [x[1] for x in results_max], '^',
#              [x[0] for x in results_var], [x[1] for x in results_var], 'o',
#              [x[0] for x in results_min], [x[1] for x in results_min], 'k--')
#     plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'])
#     plt.xlim(0.0, 1.0)
#     plt.ylim(0.0, MSE_0)
#     plt.xlabel('Rejection Fraction')
#     plt.ylabel('% > 1 GRD AE')
#     if show==True:
#         plt.savefig(os.path.join(save_path, "AE Rejection Curve using " + measure_name+'.png'), bbox_inches='tight')
#     plt.close()
#
#     AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
#     if save_path is not None:
#         with open(os.path.join(save_path, 'results.txt'), 'a') as f:
#             f.write('AE ROC using ' + measure_name + ": " + str(np.round(AUC_RR * 100.0, 1)) + '\n')
#     return AUC_RR
#
#
# def reject_PCC(targets, preds, measure, measure_name, save_path, pos_label=1, show=True):
#     if pos_label != 1:
#         measure_loc = -1.0 * measure
#     else:
#         measure_loc = measure
#     preds = np.squeeze(preds)
#     # Compute total MSE and PCC
#     error = (preds - targets) ** 2
#     P_0 = pearsonr(preds, targets)[0]
#     # print 'BASE MSE', P_0
#     # Create array
#     array = np.concatenate(
#         (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)
#
#     # Results arrays
#     results_max = [[0.0, P_0]]
#     results_var = [[0.0, P_0]]
#     results_min = [[0.0, P_0]]
#
#     optimal_ranking = array[:, 2].argsort()
#     sorted_array = array[optimal_ranking]  # Sort by error
#
#     for i in range(1, array.shape[0]):
#         x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
#         p = pearsonr(x, sorted_array[:, 1])[0]
#         # Best rejection
#         results_max.append([float(i) / float(array.shape[0]), p])
#         # Random Rejection
#         results_min.append([float(i) / float(array.shape[0]), P_0 + (1.0 - P_0) * float(i) / float(array.shape[0])])
#
#     uncertainty_ranking = array[:, 3].argsort()
#     sorted_array = array[uncertainty_ranking]  # Sort by uncertainty
#
#     for i in range(1, array.shape[0]):
#         x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
#         p = pearsonr(x, sorted_array[:, 1])[0]
#         results_var.append([float(i) / float(array.shape[0]), p])
#
#     # print 'Rho', rho(optimal_ranking, uncertainty_ranking)
#     # print 'tau', tau(optimal_ranking, uncertainty_ranking)
#     max_auc = auc([x[0] for x in results_max], [x[1] - P_0 for x in results_max], reorder=True)
#     var_auc = auc([x[0] for x in results_var], [x[1] - P_0 for x in results_var], reorder=True)
#     min_auc = auc([x[0] for x in results_min], [x[1] - P_0 for x in results_min], reorder=True)
#
#     plt.scatter([x[0] for x in results_max], [x for x in np.asarray(sorted(measure_loc, reverse=True))])
#     plt.xlim(0.0, 1.0)
#     if show:
#         plt.savefig(os.path.join(save_path, measure_name), bbox_inches='tight')
#     plt.close()
#     plt.plot([x[0] for x in results_max], [x[1] for x in results_max], 'b^',
#              [x[0] for x in results_var], [x[1] for x in results_var], 'ro',
#              [x[0] for x in results_min], [x[1] for x in results_min], 'go')
#     plt.legend(['Optimal-Rejection', 'Model-Rejection', 'Expected Random-Rejection'], loc=4)
#     plt.xlim(0.0, 1.0)
#     plt.ylim(P_0, 1.0)
#     plt.xlabel('Rejection Fraction')
#     plt.ylabel('Pearson Correlation')
#     if show:
#         plt.savefig(os.path.join(save_path, "PCC Rejection Curve using " + measure_name), bbox_inches='tight')
#     plt.close()
#
#     AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)
#     with open(os.path.join(save_path, 'results.txt'), 'a') as f:
