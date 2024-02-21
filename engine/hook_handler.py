from typing import Callable

class HookHandler:
    """Class for handling forward and backward hook"""

    def __init__(self):
        self.modules = list(self.model.named_modules())
        self.layers = {name: layer for name, layer in self.modules[1:]}
        self.forward_hook_handles = []
        self.backward_hook_handles = []

    def attach_forward_hooks(self, layers_to_hook: list, hook_fn: Callable):
        """Method to attach custom forward hooks

        Args:
          layers_to_hook [list]: list of layers to hook
          hook_fn [Callable]: custom hook_fn in during forward pass
        """
        for name, layer in self.modules:
            if name in layers_to_hook:
                handle = layer.register_forward_hook(hook_fn)
                self.forward_hook_handles.append(handle)

    def attach_backward_hooks(self, layers_to_hook: list, hook_fn: Callable):
        """Method to attach custom backward hooks

        Args:
          layers_to_hook [list]: list of layers to hook
          hook_fn [Callable]: custom hook_fn in during backward pass
        """
        for name, layer in self.modules:
            if name in layers_to_hook:
                handle = layer.register_full_backward_hook(hook_fn)
                self.backward_hook_handles.append(handle)

    def remove_hooks(self):
        """Method to remove both custom forward and backward hook"""
        for handle in self.forward_hook_handles:
            handle.remove()
        self.forward_hook_handles = []
        for handle in self.backward_hook_handles:
            handle.remove()
        self.backward_hook_handles = []
