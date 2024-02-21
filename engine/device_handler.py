import torch


def to_device(X: Any, device: str):
    """Method to put variable X to gpu if available"""
    if type(X) is list:
        return [to_device(x) for x in X]
    elif type(X) is tuple:
        return tuple(to_device(x) for x in X)
    elif type(X) is dict:
        return {k: to_device(x) for k, x in X.items()}
    elif torch.is_tensor(X):
        return X.to(device)
    else:
        return X


class device_handler:
    # def __init__(self):
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self, X: Any):
        """Method to put variable X to gpu if available"""
        if type(X) is list:
            return [self.to_device(x) for x in X]
        elif type(X) is tuple:
            return tuple(self.to_device(x) for x in X)
        elif type(X) is dict:
            return {k: self.to_device(x) for k, x in X.items()}
        elif torch.is_tensor(X):
            return X.to(self.device)
        else:
            return X
