from torchvision.datasets import ImageNet

def main(argv=None):

    path = '/home/alta/uncertainty/am969/pytorch/datasets/imagenet/tars'

    dataset1 = ImageNet(root=path, split='train', download=True)
    dataset2 = ImageNet(root=path, split='val ', download=True)

if __name__ == '__main__':
    main()
