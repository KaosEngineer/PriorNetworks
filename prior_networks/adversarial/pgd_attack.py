import foolbox
import torch
from prior_networks.models import *
import numpy as np
import os, sys

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
parser.add_argument('--concentration', type=float, default=1.0,
                    help='Concentration of non-target classes.')
parser.add_argument('--epsilon', type=float, default=0.15,
                    help='Standard Deviation for sampling Adv epsilon in training.')
parser.add_argument('--gamma', type=float, default=1.0,
                    help='Weight for OOD loss.')
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
    model.eval()

    fmodel = foolbox.models.PytorchModel(model, bound=(0,1), num_classes=ckpt['num_classes'])


if __name__ == "__main__":
    main()
