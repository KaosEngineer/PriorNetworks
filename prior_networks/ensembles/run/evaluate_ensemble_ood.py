#! /usr/bin/env python

import argparse
import sys
import os
import numpy as np

from prior_networks.ensembles.ensembles import get_ensemble_predictions
from prior_networks.ensembles.uncertainties import ensemble_uncertainties
from prior_networks.assessment.ood_detection import eval_ood_detect

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
    with open('CMDs/evaluate_ensemble_ood.txt', 'a') as f:
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

    in_labels, in_probs = get_ensemble_predictions(model_dirs,
                                                   args.source_dir,
                                                   args.n_models,
                                                   prefix='in_')
    out_labels, out_probs = get_ensemble_predictions(model_dirs,
                                                     args.source_dir,
                                                     args.n_models,
                                                     prefix='out_')

    id_uncertainties = ensemble_uncertainties(in_probs, epsilon=1e-10)
    ood_uncertainties = ensemble_uncertainties(out_probs, epsilon=1e-10)

    # Save uncertainties
    for key in id_uncertainties.keys():
        np.savetxt(os.path.join(args.output_path, key + '_id.txt'), id_uncertainties[key])
        np.savetxt(os.path.join(args.output_path, key + '_ood.txt'), ood_uncertainties[key])

    # Compute Labels
    in_domain = np.zeros_like(in_labels)
    out_domain = np.ones_like(out_labels)
    domain_labels = np.concatenate((in_domain, out_domain), axis=0)

    eval_ood_detect(domain_labels=domain_labels,
                    in_uncertainties=id_uncertainties,
                    out_uncertainties=ood_uncertainties,
                    save_path=args.output_path)


if __name__ == '__main__':
    main()
