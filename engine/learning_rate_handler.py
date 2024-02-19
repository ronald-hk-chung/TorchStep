import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class learning_rate_handler:
    def __init__(self):
        self.learning_rates = []
        self.scheduler = None
        self.is_batch_lr_scheduler = False

    def set_lr_scheduler(
        self, scheduler: torch.optim.lr_scheduler, is_batch_lr_scheduler: bool = False
    ):
        """Method to set LR scheduler

        Args:
          scheduler [torch.optim.scheduler]
          is_batch_lr_scheduler [bool]: True for batch scheduler, False for epoch scheduler
        """
        self.scheduler = scheduler
        self.is_batch_lr_scheduler = is_batch_lr_scheduler

    @staticmethod
    def make_lr_fn(
        start_lr: float, end_lr: float, num_iter: int, step_mode: str = "exp"
    ):
        """Method to generate learning rate function (internal only)"""
        if step_mode == "linear":
            factor = (end_lr / start_lr - 1) / num_iter

            def lr_fn(iteration):
                return 1 + iteration * factor

        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / num_iter

            def lr_fn(iteration):
                return np.exp(factor) ** iteration

        return lr_fn

    def set_lr(self, lr: float):
        """Method to set learning rate

        Args: lr [float]: learning rate
        """
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def lr_range_test(
        self,
        end_lr: float,
        start_lr: float | None = None,
        num_iter: int = 100,
        step_mode: str = "exp",
        alpha: float = 0.05,
        show_graph: bool = True,
    ):
        """Method to perform LR Range Test
        Reference: Leslie N. Smith 'Cyclical Learning Rates for Training Neual Networks'

        Args:
          end_lr [float]: upper boundary for the LR Range test
          start_lr [float]: lower boundary for the LR Range test, Defaults to current optimizer LR
          num_iter [int]: number of interations to move from start_lr to end_lr
          step_mode [str]: show LR range test linear or log scale, Defaults to 'exp'
          alpha [float]: alpha term for smoothed loss (smooth_loss = alpha * loss + (1-alpha) * prev_loss)
          show_graph [bool]: to show LR Range Test result in plot

        Return:
          max_grad_lr [float]: LR with maximum loss gradient (steepest)
          min_loss_lr [float]: LR with minium loss (minimum)

        """
        previous_states = {
            "model": deepcopy(self.model.state_dict()),
            "optimizer": deepcopy(self.optimizer.state_dict()),
        }
        if start_lr is not None:
            self.set_lr(start_lr)
        start_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
        lr_fn = self.make_lr_fn(start_lr, end_lr, num_iter)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        tracking = {"loss": [], "lr": []}
        iteration = 0
        while iteration < num_iter:
            for batch in self.train_dataloader:
                self.batch = self.to_device(batch)
                loss, metric = self.train_step()
                self.optimizer.zero_grad()
                loss.backward()
                tracking["lr"].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    tracking["loss"].append(loss.item())
                else:
                    prev_loss = tracking["loss"][-1]
                    smoothed_loss = alpha * loss.item() + (1 - alpha) * prev_loss
                    tracking["loss"].append(smoothed_loss)
                iteration += 1
                if iteration == num_iter:
                    break
                self.optimizer.step()
                scheduler.step()
        max_grad_idx = np.gradient(np.array(tracking["loss"])).argmin()
        min_loss_idx = np.array(tracking["loss"]).argmin()
        max_grad_lr = tracking["lr"][max_grad_idx]
        min_loss_lr = tracking["lr"][min_loss_idx]
        self.optimizer.load_state_dict(previous_states["optimizer"])
        self.model.load_state_dict(previous_states["model"])
        if show_graph:
            print(f"Max Gradient: {max_grad_lr:.2E} | Lowest Loss: {min_loss_lr:.2E}")
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.plot(tracking["lr"], tracking["loss"])
            ax.scatter(
                tracking["lr"][max_grad_idx],
                tracking["loss"][max_grad_idx],
                c="g",
                label="Max Gradient",
            )
            ax.scatter(
                tracking["lr"][min_loss_idx],
                tracking["loss"][min_loss_idx],
                c="r",
                label="Min Loss",
            )
            if step_mode == "exp":
                ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Loss")
            ax.legend()
            fig.tight_layout()
        else:
            return max_grad_lr, min_loss_lr

    def fit_one_cycle(self, epochs, max_lr=None, min_lr=None):
        """Method to perform fit one cycle polcy
        Reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
        Reference: https://arxiv.org/abs/1708.07120

        Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
        The 1cycle policy anneals the learning rate from an initial learning rate to some maximum learning rate
        and then from that maximum learning rate to some minimum learning rate

        Args:
          epochs [int]: The number of epochs to train for
          max_lr [float]: Upper learning rate boundaries in the cycle for each parameter group
          min_lr [float]: Lower learning rate boundaries in the cycle for each parameter group

          if max_lr and min_lr is not specified,
          lr_range_test will be performed
          with max_lr set to min_loss_lr and min_lr set to max_grad_lr
        """
        if max_lr is None or min_lr is None:
            max_grad_lr, min_loss_lr = self.lr_range_test(
                end_lr=1, num_iter=100, step_mode="exp", show_graph=False
            )
            if max_lr is None:
                max_lr = min_loss_lr
            if min_lr is None:
                min_lr = max_grad_lr

        print(f"Max LR: {max_lr:.1E} | Min LR: {min_lr:.1E}")
        pervious_optimizer = deepcopy(self.optimizer)
        self.set_lr(min_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            total_steps=int(len(self.train_dataloader) * epochs * 1.05),
        )
        self.set_lr_scheduler(scheduler=scheduler, is_batch_lr_scheduler=True)
        self.train(epochs=epochs)
        self.set_lr_scheduler(scheduler=None)
        self.optimizer = pervious_optimizer
