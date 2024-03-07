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

    def on_valid_loss_begin(self):
        pass

    def on_valid_loss_end(self):
        pass


class CBHandler:
    """Class for handling Callbacks"""

    def __init__(self):
        self.callback_handler = callback_handler
        self.callbacks = [
            self.PrintResults,
            self.TBWriter,
            self.SaveResults,
            self.LearningRateScheduler,
            self.GradientClipping,
        ]


class callback_handler:
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

    def on_valid_loss_begin(self):
        for callback in self.callbacks:
            callback.on_valid_loss_begin(self)

    def on_valid_loss_end(self):
        for callback in self.callbacks:
            callback.on_valid_loss_end(self)
