import context
import argparse
import sys
import os
import numpy as np

import matplotlib
import torch
import torch.nn.functional as F
from pathlib import Path

from prior_networks.assessment.misc_detection import eval_misc_detect
from prior_networks.assessment.calibration import classification_calibration
from prior_networks.assessment.rejection import eval_rejection_ratio_class
from prior_networks.priornet.dpn import dirichlet_prior_network_uncertainty
from prior_networks.ensembles.uncertainties import ensemble_uncertainties

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('models_parent_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('model_name', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('source_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('output_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--n_models', type=int, default=10,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--show', type=bool, default=True,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--overwrite', type=bool, default=False,
                               help='which orignal data is saved should be loaded')


def get_ensemble_predictions(model_dirs, experiment_path, n_models):
    """
    Get the target labels and model predictions from a txt file from all the models pointed to by list model_dirs.
    :param model_dirs: list of paths to model directories
    :param rel_labels_filepath: path to where the labels/predictions file is located within each model directory
    :return: ndarray of target labels and ndarray predictions of each model with shape [num_examples, num_models]
    """
    labels_files = map(lambda model_dir: os.path.join(model_dir, experiment_path + '/labels.txt'), model_dirs)
    prob_files = map(lambda model_dir: os.path.join(model_dir, experiment_path + '/probs.txt'), model_dirs)

    # List to store predictions from all the models considered
    all_labels, all_probs = [], []
    for labels_filepath, probs_filepath in zip(labels_files, prob_files):
        # Get the predictions from each of the models
        labels = np.loadtxt(labels_filepath, dtype=np.int32)
        probs = np.loadtxt(probs_filepath, dtype=np.float32)

        all_labels.append(labels)
        all_probs.append(probs)

    labels = np.stack(all_labels, axis=1)
    probs = np.stack(all_probs, axis=1)

    labels = np.max(np.reshape(labels, (-1, n_models)), axis=1)

    return labels, probs


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_ensemble.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_dir) and not args.overwrite:
        print(f'Directory {args.output_dir}, exists. Exiting...')
        sys.exit()
    elif os.path.isdir(args.output_dir) and args.overwrite:
        os.remove(args.output_dir + '/*')
    else:
        os.makedirs(args.output_dir)

    model_dirs = [os.path.join(args.models_parent_dir,
                               args.model_name + "{}".format(int(i))) for i in range(0, args.n_models)]
    labels, probs = get_ensemble_predictions(model_dirs, args.source_dir, args.n_models)

    mean_probs = np.mean(probs, axis=1)

    accuracy = np.mean(np.asarray(labels == np.argmax(mean_probs, axis=1), dtype=np.float32))
    with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100*(1.0-accuracy),1)} % \n')

    # Get dictionary of uncertainties.
    uncertainties = ensemble_uncertainties(probs, epsilon=1e-10)

    # Save uncertainties
    for key in uncertainties.keys():
        np.savetxt(os.path.join(args.output_path, key + '.txt'), uncertainties[key])

    # Assess Misclassification Detection
    eval_misc_detect(labels, probs, uncertainties, save_path=args.output_path, misc_positive=True)

    # Assess Calibration
    classification_calibration(labels=labels, probs=mean_probs, save_path=args.output_path)

    # Assess Rejection Performance
    eval_rejection_ratio_class(labels=labels, probs=mean_probs, uncertainties=uncertainties,
                               save_path=args.output_path)


if __name__ == '__main__':
    main()
