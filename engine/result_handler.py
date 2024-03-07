import numpy as np
from .callback_handler import Callback
import matplotlib.pyplot as plt


class ResultHandler:
    """Class for handling Printing and Saving Results"""

    def __init__(self):
        self.PrintResults = PrintResults
        self.SaveResults = SaveResults
        self.metric_keys = None
        self.progress = None
        self.batch_num = None
        self.results = {
            "train_loss": [],
            "train_metric": [],
            "valid_loss": [],
            "valid_metric": [],
        }

    def plot_loss_curve(self):
        """Method to plot loss curve"""
        plt.plot(self.results["train_loss"], label="Train Loss")
        plt.plot(self.results["valid_loss"], label="Valid Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.legend()


class PrintResults(Callback):
    def on_epoch_end(self):
        print(
            f"\r Epoch: {self.total_epochs} "
            + f"| LR: {np.array(self.learning_rates).mean():.1E} "
            + f"| train_loss: {np.around(self.train_loss, 3)} "
            + (
                f"| valid_loss: {np.around(self.valid_loss, 3)} "
                if self.valid_dataloader
                else ""
            )
        )
        if self.metric_fn:
            train_metric = (
                dict(zip(self.metric_keys, np.around(self.train_metric, 3)))
                if self.metric_keys
                else np.around(self.train_metric, 3)
            )
            print(f"train_metric: {train_metric}")
            if self.valid_dataloader:
                valid_metric = (
                    dict(zip(self.metric_keys, np.around(self.valid_metric, 3)))
                    if self.metric_keys
                    else np.around(self.valid_metric, 3)
                )
                print(f"valid_metric: {valid_metric}")

    def on_loss_begin(self):
        if isinstance(self.metric, dict):
            if self.metric_keys is None:
                self.metric_keys = list(self.metric.keys())
            self.metric = list(self.metric.values())

    def on_valid_loss_begin(self):
        if isinstance(self.metric, dict):
            if self.metric_keys is None:
                self.metric_keys = list(self.metric.keys())
            self.metric = list(self.metric.values())

    def on_loss_end(self):
        loss = np.around(self.train_loss / (self.batch_num + 1), 3)
        metric = np.around(self.train_metric / (self.batch_num + 1), 3)
        if self.metric_keys:
            metric = dict(zip(self.metric_keys, metric))
        print(
            f"\r Train Step {self.batch_num+1} / {len(self.train_dataloader)} | Loss: {loss} | Metric: {metric}",
            end="",
        )

    def on_valid_loss_end(self):
        loss = np.around(self.train_loss / (self.batch_num + 1), 3)
        metric = np.around(self.train_metric / (self.batch_num + 1), 3)
        if self.metric_keys:
            metric = dict(zip(self.metric_keys, metric))
        print(
            f"\r Valid Step {self.batch_num+1} / {len(self.valid_dataloader)} | Loss: {loss} | Metric: {metric}",
            end="",
        )


class SaveResults(Callback):
    def on_epoch_end(self):
        self.results["train_loss"].append(self.train_loss)
        self.results["train_metric"].append(self.train_metric)
        self.results["valid_loss"].append(self.valid_loss)
        self.results["valid_metric"].append(self.valid_metric)
