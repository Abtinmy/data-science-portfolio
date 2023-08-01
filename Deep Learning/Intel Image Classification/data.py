import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    class_names = train_data.classes

    n_validation = 3000
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)
    valid_idx, train_idx = indices[:n_validation], indices[n_validation:]

    val_data = torch.utils.data.Subset(train_data, indices=valid_idx)
    train_data = torch.utils.data.Subset(train_data, indices=train_idx)

    test_data = datasets.ImageFolder(test_dir, transform=transform)

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
