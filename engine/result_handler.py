import numpy as np
from .callback_handler import Callback


class ResultHandler:
    """Class for handling Printing and Saving Results"""

    def __init__(self):
        self.PrintResults = PrintResults
        self.SaveResults = SaveResults
        self.results = {
            "train_loss": [],
            "train_metric": [],
            "valid_loss": [],
            "valid_metric": [],
        }


class PrintResults(Callback):
    def on_epoch_end(self):
        print(
            f"Epoch: {self.total_epochs} "
            + f"| LR: {np.array(self.learning_rates).mean():.1E} "
            + f"| train_loss: {np.around(self.train_loss, 3)} "
            + (
                f"| valid_loss: {np.around(self.valid_loss, 3)} "
                if self.valid_dataloader
                else ""
            )
        )
        if self.metric_fn:
            if self.metric_keys:
                train_metric = dict(
                    zip(self.metric_keys, np.around(self.train_metric, 3))
                )
                valid_metric = (
                    dict(zip(self.metric_keys, np.around(self.valid_metric, 3)))
                    if self.valid_dataloader
                    else None
                )
            else:
                train_metric = np.around(self.train_metric, 3)
                valid_metric = (
                    np.around(self.valid_metric, 3) if self.valid_dataloader else None
                )
            print(f"train_metric: {train_metric}")
            if self.valid_dataloader:
                print(f"valid_metric: {valid_metric}")


class SaveResults(Callback):
    def on_epoch_end(self):
        self.results["train_loss"].append(self.train_loss)
        self.results["train_metric"].append(self.train_metric)
        self.results["valid_loss"].append(self.valid_loss)
        self.results["valid_metric"].append(self.valid_metric)
