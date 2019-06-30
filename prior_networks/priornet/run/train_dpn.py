import argparse

import torch

from prior_networks.priornet.losses import PriorNetMixedLoss, \
    DirichletKLLoss
from prior_networks.util_pytorch import tv_model_dict

parser = argparse.ArgumentParser(description='Train a Dirichlet Prior Network model using a '
                                             'standard Torchvision architecture on a Torchvision '
                                             'dataset.')
parser.add_argument('--target_concentration',
                    type=float,
                    default=1e3,
                    help='Target in-domain concentration.')
parser.add_argument('--gamma',
                    type=float,
                    default=10,
                    help='Weight for OOD loss.')
parser.add_argument('--gpu', type=int, default=0, help='Which GPU to run on.')
parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs for training.')


def main():
    args = parser.parse_args()

    # Load up the model
    model_spec = torch.load('model/model.tar')
    model = tv_model_dict[model_spec['arch']](num_classes=model_spec['n_out'], pretrained=False)
    model.load_state_dict(model_spec['model_state_dict'])

    # Check that we are training on a sensible GPU
    #TODO Make more sensible device checking code
    #TODO add multi-gpu support
    assert args.gpu <= torch.cuda.device_count()-1
    device = torch.device('cuda:'+str(args.gpu))
    model.to(device)



    criterion = PriorNetMixedLoss([DirichletKLLoss(target_concentration=args.target_concentration),
                                  DirichletKLLoss()],
                                  [1., args.gamma])


    #Save final model
    torch.save({'arch': args.arch,
                'n_in': args.n_in,
                'n_channels': args.n_channels,
                'n_out': args.n_out,
                'model_state_dict': model.state_dict()},
                'model/model.tar')

if __name__ == "__main__":
    main()