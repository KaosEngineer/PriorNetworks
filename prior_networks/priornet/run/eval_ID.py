#! /usr/bin/env python
import argparse
import os
import sys

import matplotlib
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10

from prior_networks.assessment.misc_detection import eval_misc_detect
from prior_networks.evaluation import eval_logits_on_dataset
from prior_networks.priornet.dpn import dirichlet_prior_network_uncertainty
from prior_networks.util_pytorch import load_model

matplotlib.use('agg')

commandLineParser = argparse.ArgumentParser(description='Evaluates model predictions and uncertainty '
                                                        'on in-domain test data')
commandLineParser.add_argument('dataset', type=str,
                               help='Specify name of dataset to evaluate model on.')
commandLineParser.add_argument('output_path', type=str,
                               help='Path of directory for saving model outputs.')
commandLineParser.add_argument('--batch_size', type=int, default=256,
                               help='Batch size for processing')
commandLineParser.add_argument('data_path', type=str, default='./',
                               help='Path where data is saved')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
commandLineParser.add_argument('--gpu', type=int, default=0,
                               help='Specify which GPU to evaluate on.')
commandLineParser.add_argument('--overwrite', action='store_true',
                               help='Whether to overwrite a previous run of this script')
def main(argv=None):
    args = commandLineParser.parse_args()
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_ID_eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')
    if os.path.isdir(args.output_path) and not args.overwrite:
        print('Directory', args.output_path, "exists. Exiting...")
        sys.exit()
    elif os.path.isdir(args.output_path) and args.overwrite:
        os.remove(args.output_path+'/*')
    else:
        os.makedirs(args.output_path)

    #Load the model
    model = load_model(os.path.join(args.load_path, 'model'))

    #Define dataset transforms
    data_transforms = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

    #Load dataset
    dataset = CIFAR10(root=args.data_path,
                       transform=data_transforms['val'], train=False,
                       #target_transform=target_transform(1e2, 1.0),
                       download=True)

    #Evaluate the model
    labels, logits = eval_logits_on_dataset(model=model, dataset=dataset, batch_size=args.batch_size, device=args.device)
    labels, probs, logits = labels.numpy(), F.softmax(logits).numpy(), logits.numpy()

    #Get dictionary of uncertainties.
    uncertainties = dirichlet_prior_network_uncertainty(logits)

    eval_misc_detect(labels, probs, uncertainties, save_path=args.output_path, misc_positive=True)


