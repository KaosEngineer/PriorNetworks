import torch
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from prior_networks.priornet.dpn import PriorNet
from prior_networks.util_pytorch import save_model
from prior_networks.priornet.training import TrainerWithOOD
from prior_networks.priornet.losses import PriorNetMixedLoss, \
    DirichletKLLoss
import seaborn as sns

sns.set()


def main(argv=None):
    device = torch.device('cuda')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(32),
            transforms.Pad(4,padding_mode='symmetric'),
            transforms.RandomCrop(32),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'ood': transforms.Compose([
            transforms.Resize(32),
            transforms.Pad(4,padding_mode='symmetric'),
            transforms.RandomCrop(32),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    id_dataset = CIFAR10(root='/home/malinin',
                         transform=data_transforms['train'], train=True,
                         download=True)
    test_dataset = CIFAR10(root='/home/malinin',
                           transform=data_transforms['val'], train=False,
                           download=True)
    ood_dataset = CIFAR100(root='/home/malinin',
                           transform=data_transforms['ood'], train=True,
                           download=True)
    #model = models.vgg16(pretrained=False, num_classes=10)
    # model = models.vgg16_bn(pretrained=False, num_classes=10)
    model = models.resnet18(pretrained=False, num_classes=10)
    model.to(device)

    print(len(id_dataset))

    criterion = PriorNetMixedLoss(
        [DirichletKLLoss(target_concentration=1e2),
         DirichletKLLoss()],
        [1., 10.])
    cycle_len = len(id_dataset) / 64.0 * 30.0
    trainer = TrainerWithOOD(model, criterion,
                             train_dataset=id_dataset,
                             ood_dataset=ood_dataset,
                             test_dataset=test_dataset,
                             optimizer=optim.SGD,
                             #optimizer=AdamM,
                             device=device,
                             scheduler=optim.lr_scheduler.CyclicLR,
                             optimizer_params={'lr': 1e-5, 'momentum': 0.95,
                                               'weight_decay': 1e-6},

                             scheduler_params={'base_lr': 1e-5, 'max_lr': 1e-4, 'step_size_up': cycle_len/2.0,
                                               'base_momentum': 0.86, 'max_momentum': 0.95,
                                               'step_size_down': cycle_len/2.0, 'cycle_momentum': True, 'gamma': 1.0},
                             batch_size=64)
    trainer.train(30, device=device)
    #save_model(model, './test_model')
    trainer = TrainerWithOOD(model, criterion,
                             train_dataset=id_dataset,
                             ood_dataset=ood_dataset,
                             test_dataset=test_dataset,
                             optimizer=optim.Adam,
                             device=device,
                             scheduler=optim.lr_scheduler.ExponentialLR,
                             optimizer_params={'lr': 1e-5, 'weight_decay': 1e-6},
                             scheduler_params={'gamma': 0.95},
                             batch_size=64)
    trainer.train(15, device=device)
    #torch.save(model, 'model.pt')




if __name__ == '__main__':
    main()
