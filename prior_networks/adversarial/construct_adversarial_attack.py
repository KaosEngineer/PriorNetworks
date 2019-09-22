#! /usr/bin/env python
import context
import argparse
import os
import sys
import numpy as np
import time
from PIL import Image

import torch
from pathlib import Path
from torch.utils.data import DataLoader
import foolbox
from foolbox.models import PyTorchModel
from foolbox.batch_attacks import CarliniWagnerL2Attack, EADAttack, GradientSignAttack

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
parser.add_argument('attack', choices=['CWL2', 'EAD'],
                    help='Specify name of dataset to evaluate model on.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='Batch size for processing')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUs to to run on.')
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
        os.makedirs(args.output_path/'images')

    # Check that we are using a sensible GPU
    device = select_gpu(args.gpu)

    # Load up the model
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    model.to(device)
    model.eval()

    # Wrap model with a Foolbox wrapper.
    mean = np.array([0.4914, 0.4823, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.247, 0.243, 0.261]).reshape((3, 1, 1))

    fmodel = PyTorchModel(model, bounds=(0, 1), num_classes=ckpt['num_classes'], preprocessing=(mean, std))

    # Load the evaluation data
    if args.train:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mode='train'),
                                             target_transform=None,
                                             download=True,
                                             split='train')
    else:
        dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
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

    adversarials = []
    for i, data in enumerate(loader):
        start = time.time()
        images, labels = data
        images = images.numpy()
        labels = labels.numpy()
        adversarials.extend(attack(inputs=images, labels=labels, unpack=False))
        print(f"Batch {i}/{len(loader)} took {np.round((time.time()-start) / 60.0, 1)} minutes.")

    adv_labels = np.stack([adversarial.adversarial_class for adversarial in adversarials], axis=0)
    labels = np.stack([adversarial.original_class for adversarial in adversarials], axis=0)
    distances = np.stack([adversarial.distance for adversarial in adversarials], axis=0)
    logits = np.stack([adversarial.output for adversarial in adversarials], axis=0)

    np.savetxt(args.output_path / 'labels.txt', labels, dtype=np.int32)
    np.savetxt(args.output_path / 'adv_labels.txt', adv_labels, dtype=np.int32)
    np.savetxt(args.output_path / 'logits.txt', logits, dtype=np.float32)
    np.savetxt(args.output_path / 'distances.txt', distances, dtype=np.float32)

    accuracy = np.mean(np.asarray(labels == adv_labels, dtype=np.float32))
    sr = np.mean(np.asarray(labels != adv_labels, dtype=np.float32))
    with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100*(1.0-accuracy),1)} \n')
        f.write(f'Success Rate: {np.round(100*sr, 1)} \n')

    print("Saving images to folder...")
    adversarial_images = np.stack([adversarial.perturbed for adversarial in adversarials], axis=0)
    for i, image in enumerate([np.asarray(255.0*adversarial.perturbed, dtype=np.uint8) for adversarial in adversarials]):
        print(np.max(adversarial_images), np.min(adversarial_images))
        Image.fromarray(image).save(args.output_path / f"images/{i}.png")

if __name__ == "__main__":
    main()
