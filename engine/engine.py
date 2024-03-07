"""
Contains class torchstep for train and valid step for PyTorch Model
"""

from typing import Callable
from tqdm.auto import tqdm
import numpy as np
import torch
from .device_handler import DeviceHandler
from .callback_handler import CBHandler
from .learning_rate_handler import LRHandler
from .tensorboard_handler import TBHandler
from .torchinfo_handler import TorchInfoHandler
from .result_handler import ResultHandler
from .gradient_handler import GradientHandler
from .hook_handler import HookHandler
from .checkpoint_handler import CheckPointHandler

Handles = [
    DeviceHandler,
    LRHandler,
    TBHandler,
    TorchInfoHandler,
    ResultHandler,
    GradientHandler,
    HookHandler,
    CheckPointHandler,
    CBHandler,
]


class TSEngine(*Handles):
    """TorchStep class contains a number of useful functions for Pytorch Model Training

    Args:
        model [nn.Module]: torch model
        optim [tuple[Optimizer, dict[str, float]]]: Tuple(Optimizer, params dict), example: optim=(torch.optim.Adam, {'lr': 1e-3})
        loss_fn [Callable]: loss function
        metric_fn [Callable]: metric function, Default to None
        train_dataloader [DataLoader]: train dataloader, Default to None, can be set using set_loaders()
        valid_dataloader [DataLoader]: valid dataloader, Default to None, can be set using set_loaders()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: tuple[torch.optim.Optimizer, dict[str, float]],
        loss_fn: Callable,
        metric_fn: Callable = None,
        train_dataloader: torch.utils.data.DataLoader = None,
        valid_dataloader: torch.utils.data.DataLoader = None,
    ):

        self.model = model
        self.optimizer = optim[0](params=self.model.parameters(), **optim[1])
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.total_epochs = 0
        self.batch = None
        self.train_loss = None
        self.train_metric = None
        self.valid_loss = None
        self.valid_metric = None
        self.loss = None
        self.metric = None
        for handle in Handles:
            handle.__init__(self)

    def set_optimizer(self, optim: tuple[torch.optim.Optimizer, dict[str, float]]):
        """Method to set optimizer

        Args:
          optim [tuple[torch.optim.Opimizer, dictionary of parameters]]
          Example usage: optim=(torch.optim.Adam, {'lr': 1e-3})
        """
        self.optimizer = optim[0](params=self.model.parameters(), **optim[1])

    def set_loaders(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        valid_dataloader: torch.utils.data.DataLoader = None,
    ):
        """Method to set dataloaders

        Args: train_dataloader, valid_dataloader
        """
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

    def train(self, epochs: int):
        """Method for TSEngine to run train and valid loops

        Args: epochs [int]: num of epochs to run
        """
        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            self.total_epochs += 1
            self.callback_handler.on_epoch_begin(self)
            self._train_loop()
            if self.valid_dataloader:
                self._valid_loop()
            self.callback_handler.on_epoch_end(self)

    def _train_loop(self):
        self.callback_handler.on_train_begin(self)
        self.set_train_mode()
        self.train_loss, self.train_metric = 0, 0
        for self.batch_num, batch in enumerate(self.train_dataloader):
            self.callback_handler.on_batch_begin(self)
            self.batch = self.to_device(batch)
            self.loss, self.metric = self.train_step()
            self.callback_handler.on_loss_begin(self)
            self.train_loss += np.array(self.loss.item())
            self.train_metric += np.array(self.metric)
            self.callback_handler.on_loss_end(self)
            self.loss.backward()
            self.callback_handler.on_step_begin(self)
            self.optimizer.step()
            self.callback_handler.on_step_end(self)
            self.optimizer.zero_grad()
            self.callback_handler.on_batch_end(self)
        self.train_loss /= len(self.train_dataloader)
        self.train_metric /= len(self.train_dataloader)
        self.callback_handler.on_train_end(self)

    def _valid_loop(self):
        self.callback_handler.on_valid_begin(self)
        self.set_valid_mode()
        self.valid_loss, self.valid_metric = 0, 0
        with torch.inference_mode():
            for self.batch_num, batch in enumerate(self.valid_dataloader):
                self.batch = self.to_device(batch)
                self.loss, self.metric = self.valid_step()
                self.callback_handler.on_valid_loss_begin(self)
                self.valid_loss += np.array(self.loss.item())
                self.valid_metric += np.array(self.metric)
                self.callback_handler.on_valid_loss_end(self)
        self.valid_loss /= len(self.valid_dataloader)
        self.valid_metric /= len(self.valid_dataloader)
        self.callback_handler.on_valid_end(self)

    def train_step(self):
        """Standard train step"""
        X, y = self.split_batch()
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def valid_step(self):
        """Standard valid step"""
        X, y = self.split_batch()
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def split_batch(self):
        X, *y = self.batch
        y = y[0] if len(y) == 1 else y
        return X, y

    def set_train_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.train()

    def set_valid_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.eval()

    def freeze(self, layers: list[str] = None):
        """Method to change requires_grad to False for layers

        Args: layers [list[str]]: list of layers to freeze, freeze all if None
        """
        if layers is None:
            layers = [
                name for name, module in self.model.named_modules() if "." not in name
            ]

        for layer in layers:
            for name, module in self.model.named_modules():
                if layer in name:
                    for param in module.parameters():
                        param.requires_grad = False

    def unfreeze(self, layers: list[str] = None):
        """Method to change requires_grad to True for layers

        Args: layers [list[str]]: list of layers to unfreeze, unfreeze all if None
        """
        if layers is None:
            layers = [
                name for name, module in self.model.named_modules() if "." not in name
            ]

        for layer in layers:
            for name, module in self.model.named_modules():
                if layer in name:
                    for param in module.parameters():
                        param.requires_grad = True

    def predict(self, X):
        """Method for TSEngine to predict in inference_mode"""
        X = self.to_device(X)
        self.model.eval()
        with torch.inference_mode():
            y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        self.model.train()
        return y_logits
