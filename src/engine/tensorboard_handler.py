import datetime
from torch.utils.tensorboard import SummaryWriter
from .callback_handler import Callback

class TBHandler:
    """Class for handling TensorBoard"""

    def __init__(self):
        self.writer = None
        self.TBWriter = TBWriter

    def set_tensorboard(self, name: str, folder: str = "runs"):
        """Method to set TSEngine tensorboard

        Args:
          name (str):   name of project
          folder (str): name of folder to run tensorboard logs, Defaults to 'runs'
        """
        suffix = datetime.datetime.now().strftime("%Y%m%d")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    def add_graph(self):
        """Method to add graph for TensorBoard"""
        if self.train_dataloader and self.writer:
            X, *y = next(iter(self.train_dataloader))
            X = self.to_device(X)
            self.writer.add_graph(self.model, X)


class TBWriter(Callback):
    def on_epoch_end(self):
        if self.writer:
            loss_scalars = {
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
            }
            self.writer.add_scalars(
                main_tag="loss",
                tag_scalar_dict=loss_scalars,
                global_step=self.total_epochs,
            )

            for i, train_metric in enumerate(self.train_metric):
                acc_scalars = {
                    "train_metric": self.train_metric[i],
                    "valid_metric": self.valid_metric[i],
                }
                self.writer.add_scalars(
                    main_tag=(
                        self.metric_keys[i] if self.metric_keys else f"metric_{i}"
                    ),
                    tag_scalar_dict=acc_scalars,
                    global_step=self.total_epochs,
                )
            self.writer.close()
