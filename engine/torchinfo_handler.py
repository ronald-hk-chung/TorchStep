import torchinfo
import torch

class torchinfo_handler:
    def model_info(
        self,
        col_names: list[str] = ["input_size", "output_size", "num_params", "trainable"],
        col_width: int = 20,
        row_settings: list[str] = ["var_names"],
    ):
        """Method to utilise torchinfo to shwo model summary
        Reference: https://github.com/TylerYep/torchinfo

        Args:
          col_names (Iterable[str]): Specify which columns to show in the output
            Currently supported: ("input_size",
                                  "output_size",
                                  "num_params",
                                  "params_percent",
                                  "kernel_size",
                                  "mult_adds",
                                  "trainable")
            Default: ["input_size", "output_size", "num_params", "trainable"]

          col_width (int): Width of each column. Default: 20
        """
        batch = next(iter(self.train_dataloader))
        self.batch = self.to_device(batch)
        X, y = self.split_batch()
        print(
            torchinfo.summary(
                model=self.model,
                input_data=X if torch.is_tensor(X) else list(X),
                verbose=0,
                col_names=col_names,
                col_width=col_width,
                row_settings=row_settings,
            )
        )
