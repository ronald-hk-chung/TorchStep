import torch
from typing import Any

def to_device(X: Any, device: str):
    """Method to put variable X to gpu if available"""
    if isinstance(X, list):
        return [to_device(x, device) for x in X]
    elif isinstance(X, tuple):
        return tuple(to_device(x, device) for x in X)
    elif isinstance(X, dict):
        return {k: to_device(x, device) for k, x in X.items()}
    elif torch.is_tensor(X):
        return X.to(device)
    else:
        return X

class DeviceHandler:
    """Class for handling device management"""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def to_device(self, X: Any):
        return to_device(X, self.device)
