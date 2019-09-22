#! /usr/bin/env python
import context
import argparse
import os
import sys
import numpy as np

from PIL import Image

import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

import matplotlib
import torch
import torch.nn.functional as F
from pathlib import Path

from prior_networks.assessment.misc_detection import eval_misc_detect
from prior_networks.evaluation import eval_logits_on_dataset
from prior_networks.datasets.image import construct_transforms
from prior_networks.assessment.calibration import classification_calibration
from prior_networks.assessment.rejection import eval_rejection_ratio_class
from prior_networks.priornet.dpn import dirichlet_prior_network_uncertainty
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.models.model_factory import ModelFactory
from prior_networks.adversarial.fgm import construct_fgm_attack

matplotlib.use('agg')

parser = argparse.ArgumentParser(description='Evaluates model predictions and uncertainty '
                                             'on in-domain test data')
parser.add_argument('data_path', type=str,
                    help='Path where data is saved')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='Specify name of dataset to evaluate model on.')
parser.add_argument('output_path', type=str,
                    help='Path of directory for saving model outputs.')
parser.add_argument('epsilon', type=int,
                    help='Strength perturbation in pixels 0-255')
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

from typing import Optional, Tuple


def construct_adversarial_dataset(model: nn.Module, epsilon, dataset: Dataset, batch_size: int = 128,
                                  device: Optional[torch.device] = None,
                                  num_workers: int = 4) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """
    Takes a model and an evaluation dataset, and returns the logits
    output by the model on that dataset as an array
    :param model: torch.nn.Module that outputs model logits
    :param dataset: pytorch dataset with inputs and labels
    :param batch_size: int
    :param device: device to use for evaluation
    :param num_workers: int, num. workers for the data loader
    :return: stacked torch tensor of logits returned by the model
    on that dataset, and the labels
    """
    # Set model in eval mode
    model.eval()

    testloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    logits_list = []
    labels_list = []
    adv_list = []
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            # Get inputs
            inputs, labels = data
            if device is not None:
                inputs, labels = map(lambda x: x.to(device),
                                     (inputs, labels))

                adv_inputs = construct_fgm_attack(model=model, labels=labels, inputs=inputs, epsilon=epsilon)
                model.zero_grad()
                logits = model(adv_inputs)

            logits_list.append(logits)
            labels_list.append(labels)
            adv_list.append(adv_list)

    logits = torch.cat(logits_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    adv = torch.cat(adv_list, dim=0)

    return logits.cpu(), labels.cpu(), adv.cpu()


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/construct_adversarial_dataset.cmd', 'a') as f:
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

    assert 0 < args.epsilon <= 255

    epsilon = float(args.epsilon) / 255

    # Load the in-domain evaluation data
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
                                                                            mean=,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            mode='eval'),
                                             target_transform=None,
                                             download=True,
                                             split='test')

    # Evaluate the model
    logits, labels, images = construct_adversarial_dataset(model=model,
                                                           dataset=dataset,
                                                           epsilon=epsilon,
                                                           batch_size=args.batch_size,
                                                           device=device)


    labels, probs, logits = labels.numpy(), F.softmax(logits, dim=1).numpy(), logits.numpy()
    images = images.numpy()
    print(images.shape)

    mean = np.array(DATASET_DICT[args.dataset].mean).reshape((3, 1, 1))
    std = np.array(DATASET_DICT[args.dataset].std).reshape((3, 1, 1))

    images= np.asarray((images*std + mean)*255.0, dtype=np.uint8)

    # Save model outputs
    np.savetxt(os.path.join(args.output_path, 'labels.txt'), labels)
    np.savetxt(os.path.join(args.output_path, 'probs.txt'), probs)
    np.savetxt(os.path.join(args.output_path, 'logits.txt'), logits)

    for i, image in enumerate(images):
        Image.fromarray(image).save(args.output_path / f"{i}.png")

    # Get dictionary of uncertainties.
    uncertainties = dirichlet_prior_network_uncertainty(logits)
    # Save uncertainties
    for key in uncertainties.keys():
        np.savetxt(os.path.join(args.output_path, key + '.txt'), uncertainties[key])

    if args.ood:
        sys.exit()

    nll = -np.mean(np.log(probs[np.arange(probs.shape[0]), np.squeeze(labels)] + 1e-10))

    accuracy = np.mean(np.asarray(labels == np.argmax(probs, axis=1), dtype=np.float32))
    with open(os.path.join(args.output_path, 'results.txt'), 'a') as f:
        f.write(f'Classification Error: {np.round(100 * (1.0 - accuracy), 1)} \n')
        f.write(f'NLL: {np.round(nll, 3)} \n')

    # Assess Misclassification Detection
    eval_misc_detect(labels, probs, uncertainties, save_path=args.output_path, misc_positive=True)

    # Assess Calibration
    classification_calibration(labels=labels, probs=probs, save_path=args.output_path)

    # Assess Rejection Performance
    eval_rejection_ratio_class(labels=labels, probs=probs, uncertainties=uncertainties,
                               save_path=args.output_path)


if __name__ == '__main__':
    main()
