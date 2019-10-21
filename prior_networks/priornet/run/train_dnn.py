import context
import argparse
import os
import sys
import pathlib
from pathlib import Path

import torch
from torch.utils import data
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.training import Trainer
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='In-domain dataset name.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--lr_decay', type=float, default=0.2, help='LR decay multiplies')
parser.add_argument('--lrc', action='append', type=int, help='LR decay milestones')
parser.add_argument('--model_dir', type=str, default='./',
                    help='absolute directory path where to save model and associated data.')
parser.add_argument('--dropout_rate', type=float, default=0.0,
                    help='Dropout rate if model uses it.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--gpu', type=int, action='append',
                    help='Specify which GPUa to to run on.')
parser.add_argument('--multi_gpu',
                    action='store_true',
                    help='Use multiple GPUs for training.')
parser.add_argument('--augment',
                    action='store_true',
                    help='Whether to use augmentation.')
parser.add_argument('--rotate',
                    action='store_true',
                    help='Whether to use rotation augmentation')
parser.add_argument('--jitter', type=float, default=0.0,
                    help='Specify how much random color, '
                         'hue, saturation and contrast jitter to apply')
parser.add_argument('--resume',
                    action='store_true',
                    help='Whether to resume training from checkpoint.')
parser.add_argument('--clip_norm', type=float, default=10.0,
                    help='Gradient clipping norm value.')
parser.add_argument('--checkpoint_path', type=str, default=None,
                    help='Path to where to checkpoint.')


def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_dnn.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    model_dir = Path(args.model_dir)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = model_dir / 'model'
    # Load up the model

    assert max(args.gpu) <= torch.cuda.device_count() - 1

    device = select_gpu(args.gpu)
    # Load up the model
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print('Using Multi-GPU training.')
    model.to(device)

    # Load the in-domain training and validation data
    train_dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                               transform=construct_transforms(n_in=ckpt['n_in'],
                                                                              mode='train',
                                                                              mean=DATASET_DICT[args.dataset].mean,
                                                                              std=DATASET_DICT[args.dataset].std,
                                                                              augment=args.augment,
                                                                              rotation=args.rotate,
                                                                              jitter=args.jitter),
                                               target_transform=None,
                                               download=True,
                                               split='train')

    val_dataset = DATASET_DICT[args.dataset](root=args.data_path,
                                             transform=construct_transforms(n_in=ckpt['n_in'],
                                                                            mean=DATASET_DICT[args.dataset].mean,
                                                                            std=DATASET_DICT[args.dataset].std,
                                                                            mode='eval'),
                                             target_transform=None,
                                             download=True,
                                             split='val')

    # Check that we are training on a sensible GPU



    # Set up training and test criteria
    criterion = torch.nn.CrossEntropyLoss()

    # Setup model trainer and train model
    trainer = Trainer(model=model,
                      criterion=criterion,
                      test_criterion=criterion,
                      train_dataset=train_dataset,
                      test_dataset=val_dataset,
                      optimizer=optim.SGD,
                      device=device,
                      checkpoint_path=checkpoint_path,
                      scheduler=optim.lr_scheduler.MultiStepLR,
                      optimizer_params={'lr': args.lr, 'momentum': 0.9,
                                        'nesterov': True,
                                        'weight_decay': args.weight_decay},
                      scheduler_params={'milestones': args.lrc, 'gamma': args.lr_decay},
                      batch_size=args.batch_size,
                      clip_norm=args.clip_norm)
    if args.resume:
        try:
            trainer.load_checkpoint(checkpoint_path / 'checkpoint.tar', True, True, map_location=device)
        except:
            print('No checkpoint found, training from empty model.')
            pass
    trainer.train(args.n_epochs, resume=args.resume)

    # Save final model
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
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
