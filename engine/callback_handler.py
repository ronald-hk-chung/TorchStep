import numpy as np


class Callback:
    def __init__(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

    def on_valid_begin(self):
        pass

    def on_valid_end(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_loss_begin(self):
        pass

    def on_loss_end(self):
        pass

    def on_step_begin(self):
        pass

    def on_step_end(self):
        pass


callbacks = []

class callback_handler:
    def __init__(self):
        self.callback_handler = CallbackHandler
        self.PrintResults = PrintResults
        self.TBWriter = TBWriter
        self.SaveResults = SaveResults
        self.LearningRateScheduler = LearningRateScheduler
        self.GradientClipping = GradientClipping
        self.callbacks = [
            self.PrintResults,
            self.TBWriter,
            self.SaveResults,
            self.LearningRateScheduler,
            self.GradientClipping,
        ]


class CallbackHandler:
    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin(self)

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end(self)

    def on_valid_begin(self):
        for callback in self.callbacks:
            callback.on_valid_begin(self)

    def on_valid_end(self):
        for callback in self.callbacks:
            callback.on_valid_end(self)

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin(self)

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end(self)

    def on_loss_begin(self):
        for callback in self.callbacks:
            callback.on_loss_begin(self)

    def on_loss_end(self):
        for callback in self.callbacks:
            callback.on_loss_end(self)

    def on_step_begin(self):
        for callback in self.callbacks:
            callback.on_step_begin(self)

    def on_step_end(self):
        for callback in self.callbacks:
            callback.on_step_end(self)


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


class LearningRateScheduler(Callback):
    def on_batch_end(self):
        if self.scheduler and self.is_batch_lr_scheduler:
            self.scheduler.step()
        self.learning_rates.append(self.optimizer.state_dict()["param_groups"][0]["lr"])

    def on_epoch_end(self):
        self.learning_rates = []
        if self.scheduler and not self.is_batch_lr_scheduler:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(self.valid_loss)
            else:
                self.scheduler.step()


class GradientClipping(Callback):
    def on_step_begin(self):
        if callable(self.clipping):
            self.clipping()
