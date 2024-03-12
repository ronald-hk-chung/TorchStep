import torch
from torch import nn
from .callback_handler import Callback

class GradientHandler:
    """Class for handling gradient clipping"""

    def __init__(self):
        self.clipping = None
        self.GradientClipping = GradientClipping

    def set_clip_grad_value(self, clip_value):
        """
        Method to perform Value Clipping
        Clips gradients element-wise so that they stay inside the [-clip_value, +clip_value]
        Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html
        Executed in GradientClipping Callback

        Args:
          clip_value (float): max and min gradient value
        """
        self.clipping = lambda: nn.utils.clip_grad_value_(
            self.model.parameters(), clip_value=clip_value
        )

    def set_clip_grad_norm(self, max_norm, norm_type=2):
        """
        Method to perform Norm Clipping
        Norm clipping computes the norm for all gradeints together if they were concatedated into a single vector
        if the norm exceeds teh clipping value, teh gradients are scaled down to match the desired norm
        Reference: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        Executed in GradientClipping Callback

        Args:
          max_norm (float): max norm of the gradients
          norm_type (float): type of the used p-norm. Can be 'inf' for infinity norm
        """
        self.clipping = lambda: nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=max_norm, norm_type=norm_type
        )

    def set_clip_backprop(self, clip_value):
        """
        Method to set clip gradient on the fly using backward hook (register_hook)
        clamp all grad using torch.clamp between [-clip_value, +clip_value]

        Args:
          clip_value (float): max and min gradient value

        """
        if self.clipping is None:
            self.clipping = []
        for p in self.model.parameters():
            if p.requires_grad:

                def func(grad):
                    return torch.clamp(grad, -clip_value, clip_value)

                handle = p.register_hook(func)
                self.clipping.append(handle)

    def remove_clip(self):
        """Method to remove gradient clipping in backward hook"""
        if isinstance(self.clipping, list):
            for handle in self.clipping:
                handle.remove()
        self.clipping = None


class GradientClipping(Callback):
    def on_step_begin(self):
        if callable(self.clipping):
            self.clipping()
