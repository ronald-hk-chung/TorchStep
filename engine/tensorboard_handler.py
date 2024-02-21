from torch.utils.tensorboard import SummaryWriter
import datetime


class tensorboard_handler:
    def __init__(self):
        self.writer = None

    def set_tensorboard(self, name: str, folder: str = "runs"):
        """Method to set TSEngine tensorboard

        Args:
          name [str]: name of project
          folder [str]: name of folder to run tensorboard logs, Defaults to 'runs'
        """
        suffix = datetime.datetime.now().strftime("%Y%m%d")
        self.writer = SummaryWriter(f"{folder}/{name}_{suffix}")

    def add_graph(self):
        """Method to add graph for TensorBoard"""
        if self.train_dataloader and self.writer:
            X, *y = next(iter(self.train_dataloader))
            X = self.to_device(X)
            self.writer.add_graph(self.model, X)
