import argparse
import os
import sys
import context

import torch
from prior_networks.models.model_factory import ModelFactory

parser = argparse.ArgumentParser(description='Setup a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision'
                                             ' dataset.')
parser.add_argument('destination_path', type=str,
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('n_in', type=int,
                    help='Choose size of input image. eg: 32".')
parser.add_argument('num_classes', type=int,
                    help='Choose size of number of classes.')
parser.add_argument('--arch',
                    choices=ModelFactory.MODEL_DICT.keys(),
                    default='vgg16',
                    help='Choose one of standard Torchvision architectures '
                         'to construct model, eg: "vgg16_bn".')
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
    os.mkdir('model')

    model = ModelFactory.create_model(args.arch,
                                      num_classes=args.num_classes,
                                      small_inputs=args.small_inputs,
                                      pretrained=False)

    ModelFactory.checkpoint_model(path='model/model.tar',
                                  model=model,
                                  arch=args.arch,
                                  n_channels=args.n_channels,
                                  num_classes=args.num_classes,
                                  small_inputs=args.small_inputs,
                                  n_in=args.n_in)


if __name__ == "__main__":
    main()
