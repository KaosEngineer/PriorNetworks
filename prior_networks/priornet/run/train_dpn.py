import context
import argparse
import os
import sys
import pathlib
from pathlib import Path

import torch
from torch.utils import data
from prior_networks.priornet.losses import DirichletKLLoss, DirichletKLLossJoint
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.priornet.training import TrainerWithOODJoint
from prior_networks.util_pytorch import TargetTransform
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('id_dataset', choices=DATASET_DICT.keys(),
                    help='In-domain dataset name.')
parser.add_argument('ood_dataset', choices=DATASET_DICT.keys(),
                    help='Out-of-domain dataset name.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--target_concentration', type=float, default=1e2,
                    help='Target in-domain concentration.')
parser.add_argument('--concentration', type=float, default=1.0,
                    help='Concentration of non-target classes.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Dropout rate if model uses it.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--reverse_KL', type=bool, default=True,
                    help='Whether to use forward or reverse KL. Default is to ALWAYS use reverse KL.')
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

    model_dir = Path(args.model_dir)
    # Load up the model
    ckpt = torch.load(model_dir / 'model/model.tar')
    model = ModelFactory.model_from_checkpoint(ckpt)



    # Load the in-domain training and validation data
    train_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                                  transform=construct_transforms(n_in=ckpt['n_in'],
                                                                                 mode='train',
                                                                                 augment=args.augment),
                                                  target_transform=TargetTransform(
                                                      args.target_concentration,
                                                      1.0),
                                                  download=True,
                                                  split='train')

    val_dataset = DATASET_DICT[args.id_dataset](root=args.data_path,
                                                transform=construct_transforms(n_in=ckpt['n_in'],
                                                                               mode='eval'),
                                                target_transform=None,
                                                download=True,
                                                split='val')

    if args.gamma > 0.0:
        # Load the out-of-domain training dataset
        ood_dataset = DATASET_DICT[args.ood_dataset](root=args.data_path,
                                                     transform=construct_transforms(
                                                         n_in=ckpt['n_in'],
                                                         mode='ood'),
                                                     target_transform=TargetTransform(0.0,
                                                                                      args.gamma,
                                                                                      ood=True),
                                                     download=True,
                                                     split='train')

        # Combine ID and OOD training datasets into a single dataset for
        # training (necessary for DataParallel training)
        train_dataset = data.ConcatDataset([train_dataset, ood_dataset])

    # Check that we are training on a sensible GPU
    assert args.gpu <= torch.cuda.device_count() - 1
    device = select_gpu(args.gpu)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print('Using Multi-GPU training.')
    model.to(device)

    # Set up training and test criteria
    criterion = DirichletKLLossJoint(concentration=args.concentration,
                                     reverse=args.reverse_KL)
    test_criterion = DirichletKLLoss(target_concentration=args.target_concentration,
                                     concentration=args.concentration,
                                     reverse=args.reverse_KL)

    # Setup model trainer and train model
    trainer = TrainerWithOODJoint(model=model,
                                  criterion=criterion,
                                  test_criterion=test_criterion,
                                  train_dataset=train_dataset,
                                  test_dataset=val_dataset,
                                  optimizer=optim.SGD,
                                  device=device,
                                  checkpoint_path=model_dir / 'model',
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
    ModelFactory.checkpoint_model(path=model_dir / 'model/model.tar',
                                  model=model,
                                  arch=ckpt['arch'],
                                  n_channels=ckpt['n_channels'],
                                  num_classes=ckpt['num_classes'],
                                  small_inputs=ckpt['small_inputs'],
                                  n_in=ckpt['n_in'])


if __name__ == "__main__":
    main()
