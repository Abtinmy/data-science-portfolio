from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch
import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def add_regularization(model: torch.nn.Module,
                       backbone: torch.nn.Module,
                       pred: torch.Tensor,
                       gt: torch.Tensor,
                       l1: float = 0,
                       l2: float = 0,
                       regularized_layer=None
                       ):
    l1_regularization = 0
    l2_regularization = 0
    layer_regularization = 0

    if l1 > 0:
        all_params = torch.cat([x.view(-1) for x in model.parameters()])
        l1_regularization = l1 * torch.norm(all_params, 1)

    if l2 > 0:
        all_params = torch.cat([x.view(-1) for x in model.parameters()])
        l2_regularization = l2 * torch.norm(all_params, 2)

    if regularized_layer is not None:
        params = torch.cat([x.view(-1) for x in regularized_layer.parameters()])
        layer_regularization = torch.norm(params, p=2) * 0.01

    loss = backbone(pred, gt) + l1_regularization + l2_regularization + layer_regularization
    return loss


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               l1: float = 0,
               l2: float = 0,
               regularized_layer=None) -> Tuple[float, float]:
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)

        loss = add_regularization(model, loss_fn, y_pred, y, l1, l2, regularized_layer)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device,
              l1: float = 0,
              l2: float = 0,
              regularized_layer=None
              ) -> Tuple[float, float]:
    model.eval()

    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)

            loss = add_regularization(model, loss_fn, test_pred_logits, y, l1, l2, regularized_layer)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          early_stopping: bool = False,
          l1: float = 0,
          l2: float = 0,
          regularized_layer=None) -> Dict[str, List]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }
    model.to(device)

    early_stopper = EarlyStopper(patience=3, min_delta=0.01)
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device,
                                           l1=l1,
                                           l2=l2,
                                           regularized_layer=regularized_layer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device,
                                        l1=l1,
                                        l2=l2,
                                        regularized_layer=regularized_layer)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if early_stopping:
            if early_stopper.early_stop(test_loss):
                break

    return results
