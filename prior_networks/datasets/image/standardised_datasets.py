"""
A number of classes that transform the torch vision datasets into a standard format
usable for uncertainty (e.g. out-of-domain vs. in-domain) experimentation.
"""
import context
import torchvision
import sys
from torchvision.datasets import DatasetFolder
import os
from torchvision import transforms
from PIL import Image
from torchvision.datasets.folder import *
import torchvision.datasets as datasets

split_options = ['train', 'val', 'test']


def construct_transforms(n_in: int, mode: str, mean: tuple, std: tuple, augment: bool = False, rotation: bool = False,
                         jitter: float = 0.0):
    """

    :param n_in:
    :param mode:
    :param augment:
    :param rotation:
    :param jitter:
    :return:
    """
    assert mode in ['train', 'eval', 'ood']
    assert not jitter < 0.0
    transf_list = []
    # TODO Make automatic. This is here temporaricly...
    mean = (0.4914, 0.4823, 0.4465)
    std = (0.247, 0.243, 0.261)

    if augment:
        if mode == 'eval':
            transf_list.extend([transforms.Resize(n_in, Image.BICUBIC),
                                transforms.CenterCrop(n_in)])
        elif mode == 'train':
            transf_list.extend([transforms.Resize(n_in, Image.BICUBIC),
                                transforms.Pad(4, padding_mode='reflect')])
            if rotation:
                transf_list.append(transforms.RandomRotation(degrees=15, resample=Image.BICUBIC))
            transf_list.extend([torchvision.transforms.ColorJitter(jitter, jitter, jitter, jitter),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(n_in)])
        else:
            transf_list.extend([transforms.Resize(n_in, Image.BICUBIC),
                                transforms.Pad(4, padding_mode='reflect')])
            if rotation:
                transf_list.append(transforms.RandomRotation(degrees=15, resample=Image.BICUBIC))
            transf_list.extend([torchvision.transforms.ColorJitter(jitter, jitter, jitter, jitter),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.RandomCrop(n_in)])
    else:
        transf_list.extend([transforms.Resize(n_in, Image.BICUBIC)])

    transf_list.extend([transforms.ToTensor(),
                        transforms.Normalize(mean, std)])

    return transforms.Compose(transf_list)


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class KMNIST(torchvision.datasets.KMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class EMNIST(torchvision.datasets.EMNIST):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         split='byclass',
                         target_transform=target_transform,
                         train=train)


class SEMEION(torchvision.datasets.SEMEION):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        print('SEMEION does not have standard test/train splits. '
              'Generating full dataset...')

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


class Omniglot(torchvision.datasets.Omniglot):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         background=train)


class CIFAR10(torchvision.datasets.CIFAR10):
    mean = (0.4914, 0.4823, 0.4465)
    std = (0.247, 0.243, 0.261)

    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        # (0.499, 0.484, 0.444), (0.258, 0.250, 0.269]) C10+C100

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class CIFAR100(torchvision.datasets.CIFAR100):
    mean = (0.507, 0.487, 0.441)
    std = (0.267, 0.256, 0.276)

    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        train = False
        if split == 'train':
            train = True

        super().__init__(root=root,
                         download=download,
                         transform=transform,
                         target_transform=target_transform,
                         train=train)


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, transform, target_transform, download, split):
        assert split in split_options
        if split == 'val':
            split = 'test'

        super().__init__(root=os.path.join(root, 'svhn'),
                         split=split,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, root, transform, target_transform, download, split):
        # TODO Add standard transforms for imagenet
        assert split in split_options
        if split == 'test':
            split = 'val'

        super().__init__(root=os.path.join(root, 'imagenet'),
                         split=split,
                         download=download,
                         transform=transform,
                         target_transform=target_transform)


class LSUN(torchvision.datasets.LSUN):
    def __init__(self, root, transform, target_transform, split, download=None):
        # TODO Add standard transforms for LSUN

        if download is not None:
            print('LSUN must be downloaded manually')

        assert split in split_options

        super().__init__(root=os.path.join(root, 'lsun'),
                         classes=split,
                         transform=transform,
                         target_transform=target_transform)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.JPEG')


class TinyImageNet(datasets.VisionDataset):
    mean = (0.4914, 0.4823, 0.4465)
    std = (0.247, 0.243, 0.261)

    def __init__(self, root, transform, target_transform, split, extensions=IMG_EXTENSIONS, loader=default_loader,
                 download=None):

        if download is not None:
            print('TinyImageNet must be downloaded manually')

        root = os.path.join(root, 'tiny-imagenet')

        assert split in split_options
        if split == 'test':
            split = 'val'

        super(TinyImageNet, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        if split == 'train':
            samples = make_dataset_TIM(os.path.join(self.root, 'train'),
                                       class_to_idx,
                                       extensions)
        else:
            samples = make_dataset_TIM_val(os.path.join(self.root, 'val'),
                                           class_to_idx,
                                           extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        with open(os.path.join(dir, 'wnids.txt')) as f:
            classes = [line[:-1] for line in f.readlines()]

        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


def make_dataset_TIM_val(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    with open(os.path.join(dir, 'val_annotations.txt')) as f:
        fileclass_dict = {}
        for line in f.readlines():
            line = line.split()
            fileclass_dict[line[0]] = line[1]

    d = os.path.join(dir, 'images')
    if not os.path.isdir(d):
        print(f'Directory {d} does not exist!')
    for root, _, fnames in sorted(os.walk(d)):
        for fname in sorted(fnames):
            target = fileclass_dict[fname]
            path = os.path.join(root, fname)
            if is_valid_file(path):
                item = (path, class_to_idx[target])
                images.append(item)
    return images


def make_dataset_TIM(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target + '/images')
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images
