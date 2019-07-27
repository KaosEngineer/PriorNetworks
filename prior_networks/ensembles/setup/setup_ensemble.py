import argparse
import os
import sys

import torch
from prior_networks.util_pytorch import MODEL_DICT, save_model, set_random_seeds

parser = argparse.ArgumentParser(description='Setup an ensemble of models using a '
                                             'standard Torchvision architecture on a Torchvision'
                                             ' dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('destination_path', type=str,
                    help='absolute directory path to the directory in which to save the ensemble.')
parser.add_argument('library_path', type=str,
                    help='absolute path to this library.')
parser.add_argument('arch',
                    choices=MODEL_DICT.keys(),
                    default='vgg16',
                    help='Choose one of standard Torchvision architectures '
                         'to construct model, eg: "vgg16_bn".')
parser.add_argument('n_in', type=int,
                    help='Choose size of input image. eg: 32".')
parser.add_argument('num_classes', type=int,
                    help='The number of classes in the data to be used.')
parser.add_argument('num_models', type=int, help='Number of ensemble members (models).')
parser.add_argument('--n_channels', type=int, default=3,
                    help='Choose number in image channels. Default 3 for color images.')
parser.add_argument('--small_inputs', action='store_true',
                    help='Whether model should be setup to use small inputs.')
parser.add_argument('--override_directory', action='store_true', default=False,
                    help='If the ensemble directory already exists, whether to override and write'
                         ' to that directory')


def main():
    args = parser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/setup_dpn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_path) and not args.override_directory:
        raise EnvironmentError('Destination directory exists. To override the directory run with '
                               'the --override_directory flag.')
    else:
        os.makedirs(args.destination_path)

    # Link and and create directories
    os.chdir(args.destination_path)

    os.symlink(args.data_path, 'data')
    os.symlink(args.library_path, 'prior_networks')

    for i in range(args.num_models):
        model_dir_name = f'model{i}'
        os.mkdir(model_dir_name)

        set_random_seeds(i)
        model = MODEL_DICT[args.arch](pretrained=False, num_classes=args.num_classes)

        save_model(model=model,
                   n_in=args.n_in,
                   n_channels=args.n_channels,
                   num_classes=args.num_classes,
                   arch=args.arch,
                   small_inputs=args.small_inputs,
                   path=model_dir_name)


if __name__ == "__main__":
    main()
