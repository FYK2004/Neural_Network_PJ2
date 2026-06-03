from __future__ import annotations

from pathlib import Path

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


# CIFAR-10 official statistics
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, n_items: int):
        self.dataset = dataset
        self.n_items = min(n_items, len(dataset))

    def __getitem__(self, index: int):
        return self.dataset[index]

    def __len__(self) -> int:
        return self.n_items


def _train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def _test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_cifar10_loaders(
    root: str | Path = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    n_items: int = -1,
) -> tuple[DataLoader, DataLoader]:
    root = Path(root)
    train_set = datasets.CIFAR10(
        root=root, train=True, download=True, transform=_train_transform()
    )
    test_set = datasets.CIFAR10(
        root=root, train=False, download=True, transform=_test_transform()
    )

    if n_items > 0:
        train_set = SubsetDataset(train_set, n_items)
        test_set = SubsetDataset(test_set, n_items)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
