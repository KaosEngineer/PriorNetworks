#! /usr/bin/env python
import context
import argparse
import os
import sys
import numpy as np

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import foolbox
from foolbox.models import PyTorchModel
from foolbox.batch_attacks import CarliniWagnerL2Attack, EADAttack


from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.models.model_factory import ModelFactory
from prior_networks.datasets.image import construct_transforms
from prior_networks.adversarial import AdaptiveCarliniWagnerL2Attack, AdaptiveEADAttack

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('data_path', type=str,
                    help='Path where data is saved')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='Specify name of dataset to evaluate model on.')
parser.add_argument('output_path', type=str,
                    help='Path of directory for saving model outputs.')
parser.add_argument('attack', choices=['FGSM', 'CWL2', 'EAD'],
                    help='Specify name of dataset to evaluate model on.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--gpu', type=int, default=0,
                    help='Specify which GPU to evaluate on.')
parser.add_argument('--train', action='store_true',
                    help='Whether to evaluate on the training data instead of test data')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite a previous run of this script')
parser.add_argument('--adaptive', action='store_true',
                    help='Whether to use adaptive version of adversarial attack')


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/construct_adversarial_attack.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_path) and not args.overwrite:
        print(f'Directory {args.output_path} exists. Exiting...')
        sys.exit()
    elif os.path.isdir(args.output_path) and args.overwrite:
        os.remove(args.output_path + '/*')
    else:
        os.makedirs(args.output_path)

    # Check that we are using a sensible GPU
    device = select_gpu(args.gpu)

    # Load up the model
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    # Wrap model with a Foolbox wrapper.
    fmodel = PyTorchModel(model, bounds=(-1,1), num_classes=ckpt['num_classes'])

    # Load the evaluation data
    if args.train:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mean=DATASET_DICT[args.dataset].mean,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            mode='train'),
                                             target_transform=None,
                                             download=True,
                                             split='train')
    else:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mean=DATASET_DICT[args.dataset].mean,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            mode='eval'),
                                             target_transform=None,
                                             download=True,
                                             split='test')

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)

    # Construct adversarial attack
    if args.attack == 'CWL2':
        if args.adaptive:
            attack = AdaptiveCarliniWagnerL2Attack(model=fmodel)
        else:
            attack = CarliniWagnerL2Attack(model=fmodel)
    elif args.attack == 'EAD':
        if args.adaptive:
            attack = AdaptiveEADAttack(model=fmodel)
        else:
            attack = EADAttack(model=fmodel)
    else:
        raise NotImplementedError

    n_batches = int(len(dataset) / args.batch_size)
    adversarial_images = []
    for i, data in enumerate(loader):
        images, labels = data
        images = images.numpy()
        labels = labels.numpy()

        adv = attack(inputs=images, labels=labels, unpack=True)
        #adversarial_images.append(adv)

        print(adv.shape)
        sys.exit()

    #adversarial_images = np.stack(adversarial_images, axis=0)
    #print(adversarial_images.shape)

if __name__ == "__main__":
    main()
