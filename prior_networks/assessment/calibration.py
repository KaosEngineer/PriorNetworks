import os

import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score
import seaborn as sns

sns.set()
sns.set(font_scale=1.25)


def classification_calibration(labels, probs, save_path, bins=10):
    preds = np.argmax(probs, axis=1)
    total = labels.shape[0]
    probs = np.max(probs, axis=1)
    lower = 0.0
    increment = 1.0 / bins
    upper = increment
    accs = np.zeros([bins+1], dtype=np.float32)
    gaps = np.zeros([bins+1], dtype=np.float32)
    confs = np.arange(0.0, 1.01, increment)
    ECE = 0.0
    for i in range(bins):
        ind1 = probs >= lower
        ind2 = probs < upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]
        lpreds = preds[ind]
        llabels = labels[ind]
        acc = np.mean(np.asarray(llabels == lpreds, dtype=np.float32))
        prob = np.mean(lprobs)
        if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
        gaps[i] = np.abs(acc - prob)
        accs[i] = acc
        upper += increment
        lower += increment
    ECE /= np.float(total)
    MCE = np.max(np.abs(gaps))

    accs[-1] = 1.0

    fig, ax = plt.subplots()
    plt.plot(confs, accs)
    plt.plot(confs, confs)
    plt.ylim(0.0, 1.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.xlim(0.0, 1.0)
    plt.savefig(os.path.join(save_path, 'Reliability Curve'), bbox_inches='tight')
    plt.close()
    with open(os.path.join(save_path, 'results.txt'), 'a') as f:
        f.write('ECE: ' + str(np.round(ECE * 100.0, 2)) + '\n')
        f.write('MCE: ' + str(np.round(MCE * 100.0, 2)) + '\n')


# def regression_calibration_curve(targets, preds, intervals, save_path):
#     diff = np.squeeze(abs(targets - preds))[:, np.newaxis]
#     a = np.asarray(diff < intervals, dtype=np.float32)
#     emp_frac = np.mean(a, axis=0)
#     fraction = np.arange(0.0, 1.01, 0.01)
#     plt.close()
#     #fig, ax = plt.subplots()
#     plt.plot(fraction, emp_frac)
#     plt.plot(fraction, fraction, 'k--', lw=4)
#     plt.ylim(0.0, 1.0)
#     plt.xlim(0.0, 1.0)
#     plt.ylabel('Empirical Fraction')
#     plt.xlabel('Fraction')
#     plt.legend(['Empirical Fraction', 'Fraction'])
#     plt.savefig(os.path.join(save_path, 'calibration_curve.png'), bbox_inches='tight')
#     plt.close()
#
#
# def norm_intervals(means, vars):
#     means = np.squeeze(means)
#     vars = np.squeeze(vars)
#     intervals = np.arange(0.0, 1.01, 0.01)
#     return np.asarray(
#         [[norm.interval(alpha=interval, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for mean, var in
#          zip(means, vars)])
#
#
# def t_intervals(means, vars, nus):
#     means = np.squeeze(means)
#     vars = np.squeeze(vars)
#     nus = np.squeeze(nus)
#     intervals = np.arange(0.0, 1.0, 0.01)
#     return np.asarray(
#         [[t.interval(alpha=interval, df=nu, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for mean, var, nu
#          in zip(means, vars, nus)])
#
#
# def gennorm_intervals(means, vars, betas):
#     means = np.squeeze(means)
#     vars = np.squeeze(vars)
#     betas = np.squeeze(betas)
#     intervals = np.arange(0.0, 1.0, 0.01)
#     return np.asarray(
#         [[gennorm.interval(alpha=interval, beta=beta, loc=0.0, scale=np.sqrt(var))[1] for interval in intervals] for
#          mean, var, beta in zip(means, vars, betas)])
#
#
# def norm_calibration_curve(targets, means, log_vars, save_path):
#     vars = np.exp(log_vars)
#     intervals = norm_intervals(means, vars)
#     regression_calibration_curve(targets, means, intervals, save_path)
#
#
# def td_calibration_curve(targets, means, log_vars, log_nus, save_path):
#     vars = np.exp(log_vars)
#     nus = np.exp(log_nus)
#     intervals = t_intervals(means, vars, nus)
#     regression_calibration_curve(targets, means, intervals, save_path)
#
#
# def ti_calibration_curve(targets, means, log_vars, log_kappas, log_nus, save_path):
#     vars = np.exp(log_vars)
#     kappas = np.exp(log_kappas)
#     nus = np.exp(log_nus)
#     vars = (kappas + 1) / (kappas * nus) * vars
#
#     intervals = t_intervals(means, vars, nus)
#     regression_calibration_curve(targets, means, intervals, save_path)
#
#
# def gennorm_calibration_curve(targets, means, log_vars, log_betas, save_path):
#     vars = np.exp(log_vars)
#     betas = np.exp(log_betas)
#     intervals = gennorm_intervals(means, vars, betas)
#     regression_calibration_curve(targets, means, intervals, save_path)
