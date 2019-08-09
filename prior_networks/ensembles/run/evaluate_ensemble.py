import argparse
import os
import sys

import numpy as np

from prior_networks.assessment.calibration import classification_calibration
from prior_networks.assessment.misc_detection import eval_misc_detect
from prior_networks.assessment.rejection import eval_rejection_ratio_class
from prior_networks.ensembles.ensembles import get_ensemble_predictions
from prior_networks.ensembles.uncertainties import ensemble_uncertainties

commandLineParser = argparse.ArgumentParser(description='Compute features from labels.')
commandLineParser.add_argument('models_parent_dir', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('model_name', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('source_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('output_path', type=str,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--n_models', type=int, default=10,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--show', type=bool, default=True,
                               help='which orignal data is saved should be loaded')
commandLineParser.add_argument('--overwrite', type=bool, default=False,
                               help='which orignal data is saved should be loaded')


def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate_ensemble.txt', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_path) and not args.overwrite:
        print(f'Directory {args.output_path}, exists. Exiting...')
        sys.exit()
    elif os.path.isdir(args.output_path) and args.overwrite:
        os.remove(args.output_path + '/*')
    else:
        os.makedirs(args.output_path)

    model_dirs = [os.path.join(args.models_parent_dir,
                               args.model_name + "{}".format(int(i))) for i in range(0, args.n_models)]
    labels, probs = get_ensemble_predictions(model_dirs, args.source_path, args.n_models)

    mean_probs = np.mean(probs, axis=1)

    nll = -np.mean(np.log(mean_probs[np.arange(mean_probs.shape[0]), np.squeeze(labels)] + 1e-10))

    # Get dictionary of uncertainties.
    uncertainties = ensemble_uncertainties(probs, epsilon=1e-10)

    accuracy = np.mean(np.asarray(labels == np.argmax(mean_probs, axis=1), dtype=np.float32))
    with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100*(1.0-accuracy),1)} % \n')
        f.write(f'NLL: {np.round(nll, 2)} \n')

    # Save uncertainties
    for key in uncertainties.keys():
        np.savetxt(os.path.join(args.output_path, key + '.txt'), uncertainties[key])

    # Assess Misclassification Detection
    eval_misc_detect(labels, mean_probs, uncertainties, save_path=args.output_path, misc_positive=True)

    # Assess Calibration
    classification_calibration(labels=labels, probs=mean_probs, save_path=args.output_path)

    # Assess Rejection Performance
    eval_rejection_ratio_class(labels=labels, probs=mean_probs, uncertainties=uncertainties,
                               save_path=args.output_path)


if __name__ == '__main__':
    main()
