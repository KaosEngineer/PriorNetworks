import argparse
import os
import sys

import torch
from torch.utils import data
from prior_networks.priornet.losses import DirichletKLLoss, PriorNetMixedLoss
from prior_networks.util_pytorch import model_dict, dataset_dict, select_gpu
from prior_networks.priornet.training import TrainerWithAdv
from prior_networks.util_pytorch import save_model, TargetTransform
from torch import optim
from prior_networks.datasets.image.standardised_datasets import construct_transforms

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('id_dataset', choices=dataset_dict.keys(),
                    help='In-domain dataset name.')
parser.add_argument('n_epochs', type=int,
                    help='How many epochs to train for.')
parser.add_argument('lr', type=float,
                    help='Initial learning rate.')
parser.add_argument('--target_concentration', type=float, default=1e2,
                    help='Target in-domain concentration.')
parser.add_argument('--adv_concentration', type=float, default=1.0,
                    help='Target adversarial concentration.')
parser.add_argument('--adv_noise', type=float, default=0.15,
                    help='Standard Deviation for sampling Adv epsilon in training.')
parser.add_argument('--concentration', type=float, default=1.0,
                    help='Concentration of non-target classes.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
parser.add_argument('--drop_rate', type=float, default=0.0,
                    help='Dropout rate if model uses it.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay.')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training.')
parser.add_argument('--model_load_path', type=str, default='./model',
                    help='Source where to load the model from.')
parser.add_argument('--data_path', type=str, default='./data',
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

    # Load up the model
    ckpt = torch.load('./model/model.tar')
    model = model_dict[ckpt['arch']](num_classes=ckpt['num_classes'],
                                     small_inputs=ckpt['small_inputs'])
    model.load_state_dict(ckpt['model_state_dict'])

    # Load the in-domain training and validation data
    train_dataset = dataset_dict[args.id_dataset](root=args.data_path,
                                               transform=construct_transforms(n_in=ckpt['n_in'],
                                                                              mode='train',
                                                                              augment=args.augment),
                                               target_transform=None,
                                               download=True,
                                               split='train')

    val_dataset = dataset_dict[args.id_dataset](root=args.data_path,
                                                transform=construct_transforms(n_in=ckpt['n_in'],
                                                                               mode='eval'),
                                                target_transform=None,
                                                download=True,
                                                split='val')


    # Check that we are training on a sensible GPU
    assert args.gpu <= torch.cuda.device_count()-1
    device = select_gpu(args.gpu)
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print('Using Multi-GPU training.')
    model.to(device)

    # Set up training and test criteria
    test_criterion = DirichletKLLoss(target_concentration=args.target_concentration,
                                     concentration=args.concentration,
                                     reverse=args.reverse_KL)
    adv_criterion = DirichletKLLoss(target_concentration=args.adv_concentration,
                    concentration=args.concentration,
                    reverse=args.reverse_KL)

    train_criterion = PriorNetMixedLoss([test_criterion, adv_criterion],
                                        [1.0, args.gamma])

    # Setup model trainer and train model
    trainer = TrainerWithAdv(model=model,
                             criterion=train_criterion,
                             test_criterion=test_criterion,
                             adv_criterion=test_criterion,
                             train_dataset=train_dataset,
                             adv_noise=args.adv_noise,
                             test_dataset=val_dataset,
                             optimizer=optim.SGD,
                             device=device,
                             checkpoint_path='./model',
                             scheduler=optim.lr_scheduler.MultiStepLR,
                             optimizer_params={'lr': args.lr, 'momentum': 0.9,
                                          'nesterov': True,
                                          'weight_decay': args.weight_decay},
                             scheduler_params={'milestones': [25, 40]},
                             batch_size=args.batch_size)
    trainer.train(args.n_epochs)

    # Save final model
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = model.module
    save_model(model=model,
               n_in=args.n_in,
               n_channels=args.n_channels,
               num_classes=args.num_classes,
               arch=args.arch,
               small_inputs=args.small_inputs,
               path='model')

if __name__ == "__main__":
    main()
