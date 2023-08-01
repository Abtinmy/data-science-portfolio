from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch
import data


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    model.train()

    train_loss = 0
    for batch, (X, img1, img2) in enumerate(dataloader):
        X, img1, img2 = X.to(device), img1.to(device), img2.to(device)
        pred1, pred2 = model(X)

        loss = loss_fn(pred1, img1) + loss_fn(pred2, img2)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = train_loss / len(dataloader)
    return train_loss


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    model.eval()

    test_loss = 0
    with torch.inference_mode():
        for batch, (X, img1, img2) in enumerate(dataloader):
            X, img1, img2 = X.to(device), img1.to(device), img2.to(device)
            pred1, pred2 = model(X)

            loss = loss_fn(pred1, img1) + loss_fn(pred2, img2)
            test_loss += loss.item()

    test_loss = test_loss / len(dataloader)
    return test_loss


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    results = {"train_loss": []}
    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_fn=loss_fn,
                                optimizer=optimizer,
                                device=device)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
        )

        results["train_loss"].append(train_loss)

    return results


def test_step_single(model: torch.nn.Module,
                     test_data: data.CustomDataset,
                     loss_fn: torch.nn.Module,
                     device: torch.device) -> Tuple[float, float]:
    model.eval()

    with torch.inference_mode():
        avg_img, img1, img2 = next(iter(test_data))
        avg_img, img1, img2 = avg_img.to(device), img1.to(device), img2.to(device)
        pred1, pred2 = model(avg_img)

    return avg_img[0], pred1[0], pred2[0], img1[0], img2[0]
