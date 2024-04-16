"""`SSTLearner` inherits from several sub-class handler to handler specific functions.

Below are the list of handlers:

`DeviceHandler`     -   Device management, uses `torch.device('cuda')` if gpu available
`LRHandler`         -   Method includes set Learning Rate Scheduler, LR Finder and FitOneCycle
`TBHandler`         -   Set tensorboard for training monitor
`TorchInfoHandler`  -   Method to utilise torchinfo to show model summary
`ResultHandler`     -   Record training results within SSTLearner
`GradientHandler`   -   Gradient clipping
`HookHandler`       -   Adding forward and backward hook for PyTorch training
`CheckPointHandler` -   Saving and Loading of training checkpoint
`CBHandler`         -   Adding `Callback` for default and custom functions during training

"""

from typing import Callable
from tqdm.auto import tqdm
import torch
from .device_handler import DeviceHandler
from .callback_handler import CBHandler
from .lr_handler import LRHandler
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


class SSTLearner(*Handles):
    """SSTLearner class for PyTorch training framework with utility functions
    Full documentation: https://ronald-hk-chung.github.io/sstorch/

    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        metric_fn: Callable = None,
        train_dataloader: torch.utils.data.DataLoader = None,
        valid_dataloader: torch.utils.data.DataLoader = None,
    ):
        """Initializes the SSTLearner instance

        Args:
            model (nn.Module): torch model
            loss_fn (Callable): loss function
            metric_fn (Callable): metric function, Default to None
            train_dataloader (DataLoader): train dataloader, Default to None, can be set using set_loaders()
            valid_dataloader (DataLoader): valid dataloader, Default to None, can be set using set_loaders()
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.total_epochs = 0
        self.batch = None
        self.loss = None
        self.metric = None
        for handle in Handles:
            handle.__init__(self)

    def train(self, epochs: int):
        """Method for SSTLearner to run train and valid loops

        Args:
            epochs (int): num of epochs to run
        """
        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            self.total_epochs += 1
            self.callback_handler.on_epoch_begin(self)
            self._train_loop()
            if self.valid_dataloader:
                self._valid_loop()
            self.callback_handler.on_epoch_end(self)

    def _train_loop(self):
        self.set_train_mode()
        self.callback_handler.on_train_begin(self)
        for self.batch_num, batch in enumerate(self.train_dataloader):
            self.callback_handler.on_batch_begin(self)
            self.batch = self.to_device(batch)
            self.callback_handler.on_loss_begin(self)
            self.loss, self.metric = self.train_step()
            self.callback_handler.on_loss_end(self)
            self.loss.backward()
            self.callback_handler.on_step_begin(self)
            self.optimizer.step()
            self.callback_handler.on_step_end(self)
            self.optimizer.zero_grad()
            self.callback_handler.on_batch_end(self)
        self.callback_handler.on_train_end(self)

    def _valid_loop(self):
        self.set_valid_mode()
        self.callback_handler.on_valid_begin(self)
        self.valid_loss, self.valid_metric = 0, 0
        with torch.inference_mode():
            for self.batch_num, batch in enumerate(self.valid_dataloader):
                self.batch = self.to_device(batch)
                self.callback_handler.on_valid_loss_begin(self)
                self.loss, self.metric = self.valid_step()
                self.callback_handler.on_valid_loss_end(self)
        self.callback_handler.on_valid_end(self)

    def train_step(self):
        """Standard train step"""
        X, y = self.batch
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def valid_step(self):
        """Standard valid step"""
        X, y = self.batch
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def set_train_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.train()

    def set_valid_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.eval()

    def freeze(self, layers: list[str] = None):
        """Method to change requires_grad to False for layers

        Args:
            layers (list[str]): list of layers to freeze, freeze all if None
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

        Args:
            layers (list[str]): list of layers to unfreeze, unfreeze all if None
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
