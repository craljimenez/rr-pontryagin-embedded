"""Training loop placeholder."""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = torch.device(device)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            total_loss += self.loss_fn(self.model(x), y).item()
        return total_loss / len(loader)
