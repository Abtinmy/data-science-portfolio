import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

NUM_WORKERS = os.cpu_count()


def create_dataloaders(
        transform: transforms.Compose,
        batch_size: int,
        num_workers: int = NUM_WORKERS
):
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=None
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    class_names = train_data.classes

    print("class names: ", class_names)
    print("length train data: ", len(train_data.data),
          "\nlength train targets: ", len(train_data.targets),
          "\nlength test data: ", len(test_data.data),
          "\nlength test targets: ", len(test_data.targets))

    train_dataloader = DataLoader(
        train_data,
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

    return train_dataloader, test_dataloader, class_names
