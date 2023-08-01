import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    train_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=None
    )
    class_names = train_data.classes

    train_data, val_data = torch.utils.data.random_split(
        train_data, (len(train_data.data) - 10000, 10000)
    )

    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print("class names: ", class_names)
    print("length train data: ", len(train_dataloader.dataset),
          "\nlength validation data: ", len(val_dataloader.dataset),
          "\nlength test data: ", len(test_dataloader.dataset))
    return train_dataloader, val_dataloader, test_dataloader, class_names
