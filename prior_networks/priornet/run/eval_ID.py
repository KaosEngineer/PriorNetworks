#! /usr/bin/env python
import argparse
import os
import sys

import matplotlib
import torch
from torchvision import transforms

matplotlib.use('agg')

commandLineParser = argparse.ArgumentParser(description='Evaluates model predictions and uncertainty '
                                                        'on in-domain test data')
commandLineParser.add_argument('dataset', type=str,
                               help='Specify name of dataset to evaluate model on.')
commandLineParser.add_argument('output_path', type=str,
                               help='Path of directory for saving model outputs.')
commandLineParser.add_argument('--load_path', type=str, default='./',
                               help='Specify path to model which should be loaded')
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
    model_checkpoint = torch.load(os.path.join(args.load_path, 'model/model.tar'))

    #Define dataset transforms
    data_transforms = transforms.Compose([
            transforms.Resize(model_checkpoint['n_in']),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

    #Load dataset


    #Evaluate the model





