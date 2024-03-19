import torch


class CheckPointHandler:
    """Class for handling checkpoint saving and loading"""

    def save_checkpoint(self, filename: str):
        """
        Method to save model checkpoint

        Args: 
            filename (str): filename in pt/pth of model, e.g. 'model_path/model.pt'
        """
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "results": self.results,
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        """
        Method to load model checkpoint

        Args: 
            filename (str): file path of checkpoint to load in pt/pth format, e.g. 'model_path/model.pt'
        """
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs = checkpoint["epoch"]
        self.results = checkpoint["results"]
        self.model.train()
