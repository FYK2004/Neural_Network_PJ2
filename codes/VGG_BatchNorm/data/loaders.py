from __future__ import annotations

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.datasets as datasets

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class PartialDataset(Dataset):
    def __init__(self, dataset, n_items: int = 10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return min(self.n_items, len(self.dataset))


def _train_transform(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def _test_transform(augment: bool) -> transforms.Compose:
    if augment:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_cifar_loader(
    root: str = "./data",
    batch_size: int = 128,
    train: bool = True,
    shuffle: bool = True,
    num_workers: int = 4,
    n_items: int = -1,
    augment: bool = True,
):
    transform = _train_transform(augment) if train else _test_transform(augment)
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
