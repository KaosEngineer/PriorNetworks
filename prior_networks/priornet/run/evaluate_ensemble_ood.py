#! /usr/bin/env python

import argparse
import sys

from MalLib.uncertainty.uncertainty_functions import *


commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('models_parent_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('source_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('output_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--n_models', type=int, default=10,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--show', type=bool, default=False,
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
    in_labels_files = map(lambda model_dir: os.path.join(model_dir, experiment_path+'/in_labels.txt'), model_dirs)
    in_prob_files =  map(lambda model_dir: os.path.join(model_dir, experiment_path + '/in_probs.txt'), model_dirs)

    out_labels_files = map(lambda model_dir: os.path.join(model_dir, experiment_path+'/out_labels.txt'), model_dirs)
    out_prob_files =  map(lambda model_dir: os.path.join(model_dir, experiment_path + '/out_probs.txt'), model_dirs)


    # List to store predictions from all the models considered
    all_in_labels = []
    all_in_probs  = []
    all_out_labels = []
    all_out_probs  = []

    for labels_filepath, probs_filepath in zip(in_labels_files, in_prob_files):
        # Get the predictions from each of the models
        labels = np.loadtxt(labels_filepath, dtype=np.int32)
        probs = np.loadtxt(probs_filepath, dtype=np.float64)

        all_in_labels.append(labels)
        all_in_probs.append(probs)

    for labels_filepath, probs_filepath in zip(out_labels_files, out_prob_files):
        # Get the predictions from each of the models
        labels = np.loadtxt(labels_filepath, dtype=np.int32)
        probs = np.loadtxt(probs_filepath, dtype=np.float64)

        all_out_labels.append(labels)
        all_out_probs.append(probs)


    in_labels = np.stack(all_in_labels, axis=1)
    in_probs = np.stack(all_in_probs, axis=1)
    in_labels = np.max(np.reshape(in_labels, (-1, n_models)), axis=1)

    out_labels = np.stack(all_out_labels, axis=1)
    out_probs = np.stack(all_out_probs, axis=1)
    out_labels = np.max(np.reshape(out_labels, (-1, n_models)), axis=1)

    return in_labels, in_probs, out_labels, out_probs


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_ensemble_uncertainty.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_dir) and not args.overwrite:
        print 'Directory', args.output_dir, "exists. Exiting..."
        sys.exit()
    elif os.path.isdir(args.output_dir) and args.overwrite:
        os.remove(args.output_dir+'/*')
    else:
        os.makedirs(args.output_dir)

    model_dirs=[os.path.join(args.models_parent_dir, "DNN{}".format(int(i))) for i in range(0, args.n_models)]
    in_labels, in_probs, out_labels, out_probs = get_ensemble_predictions(model_dirs, args.source_dir, args.n_models)

    in_domain = np.zeros_like(in_labels)
    out_domain = np.ones_like(out_labels)
    domain_labels = np.concatenate((in_domain, out_domain), axis=0)

    in_uncertainties  = calculate_MCDP_uncertainty(in_probs)
    out_uncertainties = calculate_MCDP_uncertainty(out_probs)
    for key in in_uncertainties.keys():
        save_path = os.path.join(args.output_dir, key+'_in.txt')
        np.savetxt(save_path, in_uncertainties[key][0])

    for key in out_uncertainties.keys():
        save_path = os.path.join(args.output_dir, key+'_out.txt')
        np.savetxt(save_path, out_uncertainties[key][0])

    plot_roc_curves(domain_labels, in_uncertainties, out_uncertainties, save_path=args.output_dir, log=False, show=args.show)
    plot_uncertainties(in_uncertainties, out_uncertainties, save_path=args.output_dir, log=False, show=args.show)

if __name__ == '__main__':
    main()