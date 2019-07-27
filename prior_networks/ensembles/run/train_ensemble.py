import context
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils import data
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.training import Trainer
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory

LOSSES_DICT = {'CrossEntropyLoss': nn.CrossEntropyLoss}

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('ensemble_dir', type=str,
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='Dataset class name.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--loss', choices=LOSSES_DICT.keys(), type=str, default='CrossEntropyLoss',
                    help='What loss to use for training the models')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Dropout rate if model uses it.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--use_gpu')
parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help='Specify which GPU to to run on.')
parser.add_argument('--multi_gpu',
                    action='store_true',
                    help='Use multiple GPUs for training.')
parser.add_argument('--augment',
                    action='store_true',
                    help='Whether to use horizontal flipping augmentation.')


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dpn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    ensemble_dir = Path(args.ensemble_dir)
    for model_dir in ensemble_dir.glob('model*'):
        # Load up the model
        ckpt = torch.load(model_dir / 'model.tar')

        model = ModelFactory.model_from_checkpoint(ckpt)

        # Load the in-domain training and validation data
        train_dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                                   transform=construct_transforms(
                                                       n_in=ckpt['n_in'],
                                                       mode='train',
                                                       augment=args.augment),
                                                   target_transform=None,
                                                   download=True,
                                                   split='train')

        val_dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                                 transform=construct_transforms(
                                                     n_in=ckpt['n_in'],
                                                     mode='eval'),
                                                 target_transform=None,
                                                 download=True,
                                                 split='val')

        # Check that we are training on a sensible GPU
        assert args.gpu <= torch.cuda.device_count() - 1
        device = select_gpu(args.gpu)
        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            print('Using Multi-GPU training.')
        model.to(device)

        # Set up training and test criteria
        try:
            criterion = LOSSES_DICT[args.loss]()
            test_criterion = LOSSES_DICT[args.loss]()
        except LookupError as e:
            raise ValueError(f'{args.loss} is not an allowed loss.') from e

        trainer = Trainer(model=model,
                          criterion=criterion,
                          train_dataset=train_dataset,
                          test_criterion=test_criterion,
                          test_dataset=val_dataset,
                          optimizer=optim.SGD,
                          scheduler=optim.lr_scheduler.MultiStepLR,
                          optimizer_params={'lr': args.lr, 'momentum': 0.9,
                                            'nesterov': True,
                                            'weight_decay': args.weight_decay},
                          scheduler_params={'milestones': [60, 120, 160], 'gamma': 0.2},
                          batch_size=args.batch_size)
        trainer.train(args.n_epochs)

        # Save final model
        if args.multi_gpu and torch.cuda.device_count() > 1:
            model = model.module
        ModelFactory.checkpoint_model(path='model/model.tar',
                                      model=model,
                                      arch=ckpt['arch'],
                                      n_channels=ckpt['n_channels'],
                                      num_classes=ckpt['num_classes'],
                                      small_inputs=ckpt['small_inputs'],
                                      n_in=ckpt['n_in'])


if __name__ == "__main__":
    main()
