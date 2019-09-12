import os

import numpy as np


def get_ensemble_predictions(model_dirs, experiment_path, n_models, prefix=None):
    """
    Get the target labels and model predictions from a txt file from all the
    models pointed to by list model_dirs.
    :param model_dirs: list of paths to model directories
    :param rel_labels_filepath: path to where the labels/predictions
           file is located within each model directory
    :return: ndarray of target labels and ndarray predictions of each
             model with shape [num_examples, num_models]
    """
    if prefix is not None:
        labels_files = map(lambda model_dir: os.path.join(model_dir,
                                                          experiment_path + f'/{prefix}labels.txt'),
                           model_dirs)
        prob_files = map(lambda model_dir: os.path.join(model_dir,
                                                        experiment_path + f'/{prefix}probs.txt'),
                         model_dirs)
    else:
        labels_files = map(lambda model_dir: os.path.join(model_dir,
                                                          experiment_path + '/labels.txt'),
                           model_dirs)
        prob_files = map(lambda model_dir: os.path.join(model_dir,
                                                        experiment_path + '/probs.txt'),
                         model_dirs)

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