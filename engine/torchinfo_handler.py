import torchinfo
import torch
from typing import Any

class TorchInfoHandler:
    def model_info(
        self,
        col_names: list[str] = ["input_size", "output_size", "num_params", "trainable"],
        col_width: int = 20,
        row_settings: list[str] = ["var_names"],
        input_data: Any = None
    ):
        """Method to utilise torchinfo to shwo model summary
        Reference: https://github.com/TylerYep/torchinfo

        Args:
            col_names (Iterable[str]): Specify which columns to show in the output
                Currently supported: ("input_size","output_size","num_params","params_percent","kernel_size","mult_adds","trainable")
                Default: ["input_size", "output_size", "num_params", "trainable"]
            
            col_width (int): Width of each column. Default: 20
            
            row_settings (Iterable[str]): Specify which features to show in a row. 
                Currently supported: ("ascii_only", "depth", "var_names")
                Default: ("var_names")

            input_data (Sequence of Tensors): 
                Arguments for the model's forward pass (dtypes inferred).
                If the forward() function takes several parameters, pass in a list of
                args or a dict of kwargs (if your forward() function takes in a dict
                as its only argument, wrap it in a list).
                Default: None

        """
        if input_data is None:
            batch = next(iter(self.train_dataloader))
            self.batch = self.to_device(batch)
            input_data, y = self.split_batch()
            input_data = input_data if torch.istensor(input_data) else [input_data]
        else:
            input_data = self.to_device(input_data)
        print(
            torchinfo.summary(
                model=self.model,
                input_data=input_data,
                verbose=0,
                col_names=col_names,
                col_width=col_width,
                row_settings=row_settings,
            )
        )
