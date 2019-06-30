import argparse
import os
import sys

import torch

from prior_networks.util_pytorch import tv_model_dict

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
                    choices=tv_model_dict.keys(),
                    default='vgg16',
                    help='Choose one of standard Torchvision architectures '
                         'to construct model, eg: "vgg16_bn".')
parser.add_argument('n_in', type=int,
                    help='Choose size of input image. eg: 32".')
parser.add_argument('n_out', type=int,
                    help='Choose size of number of classes.')
parser.add_argument('--n_channels', type=int, default=3,
                    help='Choose number in image channels. Default 3 for color images.')

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
    os.chdir(args.destination_dir)
    os.mkdir('model')
    os.symlink(args.data_path, 'data')
    os.symlink(args.library_path, 'unslib')

    model = tv_model_dict[args.arch](pretrained=False, num_classes=args.n_out)

    torch.save({'arch': args.arch,
                'n_in': args.n_in,
                'n_channels': args.n_channels,
                'n_out': args.n_out,
                'model_state_dict': model.state_dict()},
                'model/model.tar')


if __name__ == "__main__":
    main()