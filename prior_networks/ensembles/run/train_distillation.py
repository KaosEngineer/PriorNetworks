import context
import argparse
import os
import sys
import pathlib
from pathlib import Path

import torch
from torch.utils import data
from prior_networks.util_pytorch import DATASET_DICT, select_gpu
from prior_networks.ensembles.training import TrainerDistillation
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms
from prior_networks.models.model_factory import ModelFactory

from prior_networks.ensembles.ensemble_dataset import EnsembleDataset
from prior_networks.ensembles.losses import DirichletEnDDLoss, EnDLoss
from prior_networks.ensembles.training import LRTempScheduler

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('data_path', type=str,
                    help='absolute path to training data.')
parser.add_argument('dataset', choices=DATASET_DICT.keys(),
                    help='In-domain dataset name.')
parser.add_argument('ensemble_path', type=str,
                    help='Path to where ensemble is stored.')
parser.add_argument('model', type=str,
                    help='Name of model. Ex: DNN.')
# parser.add_argument('folder', type=str,
#                     help='Folder where model outputs are stored.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('temperature', type=float, help='Initial Temperature used for distillation,')
parser.add_argument('tdecay_epoch', type=int,  help='When to begin annealing temperature.')
parser.add_argument('tdecay_length', type=int, help='How slowly to aneal temperature.')
parser.add_argument('--n_models', type=int, default=10, help='Number of models from ensemble to use.')
parser.add_argument('--min_temperature', type=float, help='Minimum Temperature used for distillation,')
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
                    help='Specify which GPUs to to run on.')
# parser.add_argument('--multi_gpu',
#                     action='store_true',
#                     help='Use multiple GPUs for training.')
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
parser.add_argument('--endd',
                    action='store_true',
                    help='Whether to do Ensemble Distribution Distillation.')
parser.add_argument('--ood',
                    action='store_true',
                    help='Whether to use auxilliary "OOD" data on which ensemble is diverse.')
parser.add_argument('--ood_dataset', choices=DATASET_DICT.keys(), default='TIM',
                    help='OOD dataset name.')
parser.add_argument('--ood_folder', type=str, default=None,
                    help='OOD dataset name.')

def main():
    args = parser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_train_distillation.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    model_dir = Path(args.model_dir)

    # Check that we are training on a sensible GPU
    assert map(args.gpu) <= torch.cuda.device_count() - 1

    device = select_gpu(args.gpu)
    # Load up the model
    ckpt = torch.load(model_dir / 'model/model.tar', map_location=device)
    model = ModelFactory.model_from_checkpoint(ckpt)
    if len(args.gpu) > 1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        print('Using Multi-GPU training.')
    model.to(device)

    # Load the in-domain training and validation data
    train_dataset_class = DATASET_DICT[args.dataset]
    train_dataset_parameters = {'root': args.data_path,
                                'transform': construct_transforms(n_in=ckpt['n_in'],
                                                                  mode='train',
                                                                  mean=DATASET_DICT[args.dataset].mean,
                                                                  std=DATASET_DICT[args.dataset].std,
                                                                  augment=args.augment,
                                                                  rotation=args.rotate,
                                                                  jitter=args.jitter),
                                'target_transform': None,
                                'download': True,
                                'split': 'train'}
    train_dataset = EnsembleDataset(dataset=train_dataset_class,
                                    dataset_parameters=train_dataset_parameters,
                                    ensemble_path=args.ensemble_path,
                                    model_dirs=args.model,
                                    n_models=args.n_models,
                                    folder='train')

    val_dataset_class = DATASET_DICT[args.dataset]
    val_dataset_parameters = {'root': args.data_path,
                              'transform': construct_transforms(n_in=ckpt['n_in'],
                                                                mean=DATASET_DICT[args.dataset].mean,
                                                                std=DATASET_DICT[args.dataset].std,
                                                                mode='eval'),
                              'target_transform': None,
                              'download': True,
                              'split': 'val'}
    val_dataset = EnsembleDataset(dataset=val_dataset_class,
                                  dataset_parameters=val_dataset_parameters,
                                  ensemble_path=args.ensemble_path,
                                  model_dirs=args.model,
                                  n_models=args.n_models,
                                  folder='eval')

    if args.ood:
        assert args.ood_folder is not None
        ood_dataset_class = DATASET_DICT[args.ood_dataset]
        ood_dataset_parameters = {'root': args.data_path,
                                  'transform': construct_transforms(n_in=ckpt['n_in'],
                                                                    mean=DATASET_DICT[args.dataset].mean,
                                                                    std=DATASET_DICT[args.dataset].std,
                                                                    mode='train'),
                                  'target_transform': None,
                                  'download': True,
                                  'split': 'train'}
        ood_dataset = EnsembleDataset(dataset=ood_dataset_class,
                                      dataset_parameters=ood_dataset_parameters,
                                      ensemble_path=args.ensemble_path,
                                      model_dirs=args.model,
                                      n_models=args.n_models,
                                      folder=args.ood_folder)

        train_dataset = data.ConcatDataset([train_dataset, ood_dataset])

    # Set up training and test criteria
    test_criterion = torch.nn.CrossEntropyLoss()
    if args.endd:
        train_criterion = DirichletEnDDLoss()
    else:
        train_criterion = EnDLoss()

    # Setup model trainer and train model
    trainer = TrainerDistillation(model=model,
                                  criterion=train_criterion,
                                  test_criterion=test_criterion,
                                  train_dataset=train_dataset,
                                  test_dataset=val_dataset,
                                  optimizer=optim.SGD,
                                  device=device,
                                  checkpoint_path=model_dir / 'model',
                                  scheduler=optim.lr_scheduler.MultiStepLR,
                                  temp_scheduler=LRTempScheduler,
                                  optimizer_params={'lr': args.lr,
                                                    'momentum': 0.9,
                                                    'nesterov': True,
                                                    'weight_decay': args.weight_decay},
                                  scheduler_params={'milestones': args.lrc,
                                                    'gamma': args.lr_decay},
                                  temp_scheduler_params={'init_temp': args.temperature,
                                                         'decay_epoch': args.tdecay_epoch,
                                                         'decay_length': args.tdecay_length},
                                  batch_size=args.batch_size)
    if args.resume:
        trainer.load_checkpoint(model_dir / 'model/checkpoint.tar', True, True, True, map_location=device)
    trainer.train(args.n_epochs, resume=args.resume)

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
