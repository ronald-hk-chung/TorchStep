import numpy as np
from .callback_handler import Callback
import matplotlib.pyplot as plt


class ResultHandler:
    """Class for handling Printing and Saving Results"""

    def __init__(self):
        self.verbose = 2
        self.train_loss = None
        self.train_metric = None
        self.valid_loss = None
        self.valid_metric = None
        self.PrintResults = PrintResults
        self.SaveResults = SaveResults
        self.metric_keys = None
        self.batch_num = None
        self.results = {
            'lr': [],
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
    def on_train_begin(self):
        # resetting train_loss and train_metric
        self.train_loss, self.train_metric = 0, 0

    def on_valid_begin(self):
        # resetting train_loss and train_metric
        self.valid_loss, self.valid_metric = 0, 0

    def on_epoch_end(self):
        if self.verbose > 0:
            print(
                f"\rEpoch: {self.total_epochs} "
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
            print("-" * 100)

    def on_loss_end(self):
        if isinstance(self.metric, dict):
            if self.metric_keys is None:
                self.metric_keys = list(self.metric.keys())
            self.metric = list(self.metric.values())
        self.train_loss += np.array(self.loss.item())
        self.train_metric += np.array(self.metric)
        loss = np.around(self.loss.item(), 3)
        metric = np.around(self.metric, 3)
        avg_loss = np.around(self.train_loss / (self.batch_num + 1), 3)
        avg_metric = np.around(self.train_metric / (self.batch_num + 1), 3)
        if self.metric_keys:
            metric = dict(zip(self.metric_keys, metric))
            avg_metric = dict(zip(self.metric_keys, avg_metric))
        if self.verbose > 1:
            print(
                f"\rTrain Step {self.batch_num+1}/{len(self.train_dataloader)} | Loss(Cur/Avg): {loss} / {avg_loss} | Metric(Cur/Avg): {metric} / {avg_metric}",
                end="",
            )

    def on_valid_loss_end(self):
        if isinstance(self.metric, dict):
            if self.metric_keys is None:
                self.metric_keys = list(self.metric.keys())
            self.metric = list(self.metric.values())
        self.valid_loss += np.array(self.loss.item())
        self.valid_metric += np.array(self.metric)
        loss = np.around(self.loss.item(), 3)
        metric = np.around(self.metric, 3)
        avg_loss = np.around(self.valid_loss / (self.batch_num + 1), 3)
        avg_metric = np.around(self.valid_metric / (self.batch_num + 1), 3)
        if self.metric_keys:
            metric = dict(zip(self.metric_keys, metric))
            avg_metric = dict(zip(self.metric_keys, avg_metric))
        if self.verbose > 1:
            print(
                f"\rValid Step {self.batch_num+1}/{len(self.valid_dataloader)} | Loss(Cur/Avg): {loss} / {avg_loss} | Metric(Cur/Avg): {metric} / {avg_metric}",
                end="",
            )

    def on_train_end(self):
        self.train_loss /= len(self.train_dataloader)
        self.train_metric /= len(self.train_dataloader)

    def on_valid_end(self):
        self.valid_loss /= len(self.valid_dataloader)
        self.valid_metric /= len(self.valid_dataloader)


class SaveResults(Callback):
    def on_epoch_end(self):
        self.results['lr'].append(np.array(self.learning_rate).mean())
        self.results["train_loss"].append(self.train_loss)
        self.results["train_metric"].append(self.train_metric)
        self.results["valid_loss"].append(self.valid_loss)
        self.results["valid_metric"].append(self.valid_metric)
