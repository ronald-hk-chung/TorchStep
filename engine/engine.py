"""
Contains class torchstep for train and valid step for PyTorch Model
"""

import random
from copy import deepcopy
from typing import Any, Callable
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from .device_handler import device_handler
from .callback_handler import callback_handler
from .learning_rate_handler import learning_rate_handler
from .tensorboard_handler import tensorboard_handler


ENGINES = [learning_rate_handler, callback_handler, tensorboard_handler, device_handler]


class TSEngine(*ENGINES):
    """
    TorchStep class contains a number of useful functions for Pytorch Model Training
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

        self.model = deepcopy(model)
        self.optimizer = self.set_optimizer(optim)
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.metric_keys = None
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # self.writer = None
        # self.scheduler = None
        # self.is_batch_lr_scheduler = False
        self.clipping = None
        self.results = {
            "train_loss": [],
            "train_metric": [],
            "valid_loss": [],
            "valid_metric": [],
        }
        # self.learning_rates = []
        self.total_epochs = 0
        self.modules = list(self.model.named_modules())
        self.layers = {name: layer for name, layer in self.modules[1:]}
        self.forward_hook_handles = []
        self.backward_hook_handles = []
        # self.callbacks = [
        #     self.SaveResults,
        #     self.PrintResults,
        #     self.TBWriter,
        #     self.LearningRateScheduler,
        #     self.GradientClipping,
        # ]
        self.batch = None
        self.train_loss = None
        self.train_metric = None
        self.valid_loss = None
        self.valid_metric = None
        self.loss = None
        self.metric = None
        for engine in ENGINES:
            engine.__init__(self)

    def set_train_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.train()

    def set_valid_mode(self):
        """Method to set mode of model in _train_loop"""
        self.model.eval()

    def set_optimizer(self, optim: tuple[torch.optim.Optimizer, dict[str, float]]):
        """Method to set optimizer

        Args:
          optim [tuple[torch.optim.Opimizer, dictionary of parameters]]
          Example usage: optim=(torch.optim.Adam, {'lr': 1e-3})
        """
        optimizer = optim[0](params=self.model.parameters(), **optim[1])
        return optimizer

    @staticmethod
    def set_seed(seed=42):
        """Function to set random seed for torch, numpy and random

        Args: seed [int]: random_seed
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

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

    def split_batch(self):
        X, *y = self.batch
        y = y[0] if len(y) == 1 else y
        return X, y

    def _train_loop(self):
        self.callback_handler.on_train_begin(self)
        self.set_train_mode()
        train_loss, train_metric = 0, 0
        for batch in tqdm(self.train_dataloader, desc="Train Step", leave=False):
            self.callback_handler.on_batch_begin(self)
            self.batch = self.to_device(batch)
            self.callback_handler.on_loss_begin(self)
            self.loss, self.metric = self.train_step()
            train_loss += np.array(self.loss.item())
            train_metric += np.array(
                list(self.metric.values()) if type(self.metric) is dict else self.metric
            )
            self.callback_handler.on_loss_end(self)
            self.loss.backward()
            self.callback_handler.on_step_begin(self)
            self.optimizer.step()
            self.callback_handler.on_step_end(self)
            self.optimizer.zero_grad()
            self.callback_handler.on_batch_end(self)
        train_loss /= len(self.train_dataloader)
        train_metric /= len(self.train_dataloader)
        self.metric_keys = list(metric.keys()) if type(metric) is dict else None
        self.callback_handler.on_train_end(self)
        return train_loss, train_metric

    def train_step(self):
        """Standard train step"""
        X, y = self.split_batch()
        # X, *y = self.batch
        # y = y[0] if len(y) == 1 else y
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def _valid_loop(self):
        self.callback_handler.on_valid_begin(self)
        self.set_valid_mode()
        valid_loss, valid_metric = 0, 0
        with torch.inference_mode():
            for batch in tqdm(self.valid_dataloader, desc="Valid Step", leave=False):
                self.batch = self.to_device(batch)
                loss, metric = self.valid_step()
                valid_loss += np.array(loss.item())
                valid_metric += np.array(
                    list(metric.values()) if type(metric) is dict else metric
                )
        valid_loss /= len(self.valid_dataloader)
        valid_metric /= len(self.valid_dataloader)
        self.callback_handler.on_valid_end(self)
        return valid_loss, valid_metric

    def valid_step(self):
        """Standard valid step"""
        X, y = self.split_batch()
        y_logits = self.model(X) if torch.is_tensor(X) else self.model(*X)
        loss = self.loss_fn(y_logits, y)
        metric = self.metric_fn(y_logits, y) if self.metric_fn else 0
        return loss, metric

    def train(self, epochs: int):
        """Method for TSEngine to run train and valid loops

        Args: epochs [int]: num of epochs to run
        """
        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            self.total_epochs += 1
            self.callback_handler.on_epoch_begin(self)
            self.train_loss, self.train_metric = self._train_loop()
            if self.valid_dataloader:
                self.valid_loss, self.valid_metric = self._valid_loop()
            else:
                self.valid_loss, self.valid_metric = None, None
            self.callback_handler.on_epoch_end(self)

    def save_checkpoint(self, filename: str):
        """Method to save model checkpoint

        Args: filename [str]: filename in pt/pth of model, e.g. 'model_path/model.pt'
        """
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "results": self.results,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        """Method to load model checkpoint

        Args: file path of checkpoint to load in pt/pth format, e.g. 'model_path/model.pt'
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.results = checkpoint["results"]
        self.model.train()

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

    def plot_loss_curve(self):
        """Method to plot loss curve"""
        plt.plot(self.results["train_loss"], label="Train Loss")
        plt.plot(self.results["valid_loss"], label="Valid Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()

    def set_clip_grad_value(self, clip_value):
        """Method to perform Value Clipping
        Clips gradietns element-wise so that they stay inside the [-clip_value, +clip_value]
        Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        Executed in GradientClipping Callback
        Args:
          clip_value [float]: max and min gradient value
        """
        self.clipping = lambda: nn.utils.clip_grad_value_(
            self.model.parameters(), clip_value=clip_value
        )

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        """Method to perform Norm Clipping
        Norm clipping computes the norm for all gradeints together if they were concatedated into a single vector
        if the norm exceeds teh clipping value, teh gradients are scaled down to match the desired norm
        Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        Executed in GradientClipping Callback

        Args:
          max_norm [float]: max norm of the gradients
          norm_type [float]: type of the used p-norm. Can be 'inf' for infinity norm
        """
        self.clipping = lambda: nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm, norm_type=norm_type
        )

    def set_clip_backprop(self, clip_value):
        """Method to set clip gradient on the fly using backward hook (register_hook)
        clamp all grad using torch.clamp between [-clip_value, +clip_value]

        Args:
          clip_value [float]: max and min gradient value

        """
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:

                def func(grad):
                    return torch.clamp(grad, -clip_value, clip_value)

                handle = p.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        """Method to remove gradient clipping in backward hook"""
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None

    def attach_forward_hooks(self, layers_to_hook, hook_fn):
        """Method to attach custom forward hooks

        Args:
          layers_to_hook [list]: list of layers to hook
          hook_fn [Callable]: custom hook_fn in during forward pass
        """
        for name, layer in self.modules:
            if name in layers_to_hook:
                handle = layer.register_forward_hook(hook_fn)
                self.forward_hook_handles.append(handle)

    def attach_backward_hooks(self, layers_to_hook, hook_fn):
        """Method to attach custom backward hooks

        Args:
          layers_to_hook [list]: list of layers to hook
          hook_fn [Callable]: custom hook_fn in during backward pass
        """
        for name, layer in self.modules:
            if name in layers_to_hook:
                handle = layer.register_full_backward_hook(hook_fn)
                self.backward_hook_handles.append(handle)

    def remove_hooks(self):
        """Method to remove both custom forward and backward hook"""
        for handle in self.forward_hook_handles:
            handle.remove()
        self.forward_hook_handles = []
        for handle in self.backward_hook_handles:
            handle.remove()
        self.backward_hook_handles = []

    # class Callback:
    #     def __init__(self):
    #         pass

    #     def on_train_begin(self):
    #         pass

    #     def on_train_end(self):
    #         pass

    #     def on_valid_begin(self):
    #         pass

    #     def on_valid_end(self):
    #         pass

    #     def on_epoch_begin(self):
    #         pass

    #     def on_epoch_end(self):
    #         pass

    #     def on_batch_begin(self):
    #         pass

    #     def on_batch_end(self):
    #         pass

    #     def on_loss_begin(self):
    #         pass

    #     def on_loss_end(self):
    #         pass

    #     def on_step_begin(self):
    #         pass

    #     def on_step_end(self):
    #         pass

    # class callback_handler:
    #     def on_train_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_train_begin(self)

    #     def on_train_end(self):
    #         for callback in self.callbacks:
    #             callback.on_train_end(self)

    #     def on_valid_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_valid_begin(self)

    #     def on_valid_end(self):
    #         for callback in self.callbacks:
    #             callback.on_valid_end(self)

    #     def on_epoch_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_epoch_begin(self)

    #     def on_epoch_end(self):
    #         for callback in self.callbacks:
    #             callback.on_epoch_end(self)

    #     def on_batch_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_batch_begin(self)

    #     def on_batch_end(self):
    #         for callback in self.callbacks:
    #             callback.on_batch_end(self)

    #     def on_loss_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_loss_begin(self)

    #     def on_loss_end(self):
    #         for callback in self.callbacks:
    #             callback.on_loss_end(self)

    #     def on_step_begin(self):
    #         for callback in self.callbacks:
    #             callback.on_step_begin(self)

    #     def on_step_end(self):
    #         for callback in self.callbacks:
    #             callback.on_step_end(self)

    # class PrintResults(Callback):
    #     def on_epoch_end(self):
    #         print(
    #             f"Epoch: {self.total_epochs} "
    #             + f"| LR: {np.array(self.learning_rates).mean():.1E} "
    #             + f"| train_loss: {np.around(self.train_loss, 3)} "
    #             + (
    #                 f"| valid_loss: {np.around(self.valid_loss, 3)} "
    #                 if self.valid_dataloader
    #                 else ""
    #             )
    #         )
    #         if self.metric_fn:
    #             if self.metric_keys:
    #                 train_metric = dict(
    #                     zip(self.metric_keys, np.around(self.train_metric, 3))
    #                 )
    #                 valid_metric = (
    #                     dict(zip(self.metric_keys, np.around(self.valid_metric, 3)))
    #                     if self.valid_dataloader
    #                     else None
    #                 )
    #             else:
    #                 train_metric = np.around(self.train_metric, 3)
    #                 valid_metric = (
    #                     np.around(self.valid_metric, 3)
    #                     if self.valid_dataloader
    #                     else None
    #                 )
    #             print(f"train_metric: {train_metric}")
    #             if self.valid_dataloader:
    #                 print(f"valid_metric: {valid_metric}")

    # class TBWriter(Callback):
    #     def on_epoch_end(self):
    #         if self.writer:
    #             loss_scalars = {
    #                 "train_loss": self.train_loss,
    #                 "valid_loss": self.valid_loss,
    #             }
    #             self.writer.add_scalars(
    #                 main_tag="loss",
    #                 tag_scalar_dict=loss_scalars,
    #                 global_step=self.total_epochs,
    #             )

    #             for i, train_metric in enumerate(self.train_metric):
    #                 acc_scalars = {
    #                     "train_metric": self.train_metric[i],
    #                     "valid_metric": self.valid_metric[i],
    #                 }
    #                 self.writer.add_scalars(
    #                     main_tag=(
    #                         self.metric_keys[i] if self.metric_keys else f"metric_{i}"
    #                     ),
    #                     tag_scalar_dict=acc_scalars,
    #                     global_step=self.total_epochs,
    #                 )
    #             self.writer.close()

    # class SaveResults(Callback):
    #     def on_epoch_end(self):
    #         self.results["train_loss"].append(self.train_loss)
    #         self.results["train_metric"].append(self.train_metric)
    #         self.results["valid_loss"].append(self.valid_loss)
    #         self.results["valid_metric"].append(self.valid_metric)

    # class LearningRateScheduler(Callback):
    #     def on_batch_end(self):
    #         if self.scheduler and self.is_batch_lr_scheduler:
    #             self.scheduler.step()
    #         self.learning_rates.append(
    #             self.optimizer.state_dict()["param_groups"][0]["lr"]
    #         )

    #     def on_epoch_end(self):
    #         self.learning_rates = []
    #         if self.scheduler and not self.is_batch_lr_scheduler:
    #             if isinstance(
    #                 self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    #             ):
    #                 self.scheduler.step(self.valid_loss)
    #             else:
    #                 self.scheduler.step()

    # class GradientClipping(Callback):
    #     def on_step_begin(self):
    #         if callable(self.clipping):
    #             self.clipping()
