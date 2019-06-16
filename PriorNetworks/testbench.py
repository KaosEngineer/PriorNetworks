import torch
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from PriorNetworks.training import train_procedure_with_ood
from PriorNetworks.priornet.dpn import PriorNet
import seaborn as sns
sns.set()



def main(argv=None):
    device = torch.device('cuda')

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(32),
            transforms.RandomAffine(degrees=15,
                                    translate=(0.125,0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'ood': transforms.Compose([
            transforms.Resize(32),
            transforms.RandomAffine(degrees=15,
                                    translate=(0.125,0.125)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


    id_dataset = CIFAR10(root='/home/malinin', transform=data_transforms['train'], train=True, download=True)
    test_dataset = CIFAR10(root='/home/malinin', transform=data_transforms['val'], train=False, download=True)
    ood_dataset = CIFAR100(root='/home/malinin', transform=data_transforms['ood'], train=True, download=True)
    #model = models.vgg16(pretrained=False, num_classes=10)
    model = models.vgg16_bn(pretrained=False, num_classes=10)
    model.to(device)


    train_procedure_with_ood(model=model,
                             train_dataset=id_dataset,
                             ood_dataset=ood_dataset,
                             test_dataset=test_dataset,
                             n_epochs=20,
                             batch_size=50,
                             lr=5e-5,
                             weight_decay=1e-6,
                             device=device,
                             lr_decay=1.0,
                             gamma=10.0)

if __name__ == '__main__':
    main()