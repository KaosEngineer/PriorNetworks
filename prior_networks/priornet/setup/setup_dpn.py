import argparse
import os
import sys

import torch
from prior_networks.util_pytorch import model_dict, save_model

parser = argparse.ArgumentParser(description='Setup a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision'
                                             ' dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to data?')
parser.add_argument('destination_path', type=str,
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('library_path', type=str,
                    help='absolute path to this library')
parser.add_argument('arch',
                    choices=model_dict.keys(),
                    default='vgg16',
                    help='Choose one of standard Torchvision architectures '
                         'to construct model, eg: "vgg16_bn".')
parser.add_argument('n_in', type=int,
                    help='Choose size of input image. eg: 32".')
parser.add_argument('num_classes', type=int,
                    help='Choose size of number of classes.')
parser.add_argument('--n_channels', type=int, default=3,
                    help='Choose number in image channels. Default 3 for color images.')
parser.add_argument('--small_inputs', action='store_true',
                    help='Whether model should be setup to use small inputs.')


def main():
    args = parser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/setup_dpn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_path):
        print('destination directory exists. Exiting...')
    else:
        os.makedirs(args.destination_path)

    # Link and and create directories
    os.chdir(args.destination_path)
    os.mkdir('model')
    os.symlink(args.data_path, 'data')
    os.symlink(args.library_path, 'prior_networks')

    model = model_dict[args.arch](pretrained=False, num_classes=args.num_classes)

    save_model(model=model,
               n_in=args.n_in,
               n_channels=args.n_channels,
               num_classes=args.num_classes,
               arch=args.arch,
               small_inputs=args.small_inputs,
               path='model')


if __name__ == "__main__":
    main()
