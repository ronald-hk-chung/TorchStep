from itertools import product as product
from math import sqrt as sqrt
import torch


class PriorBox:
    """Compute priorbox corrdinates in CXCYWH for for each source feature map"""

    def __init__(self):
        self.image_size = 300
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.num_priors = len(self.aspect_ratios)
        self.variance = [0.1, 0.2]
        self.min_sizes = [21, 45, 99, 153, 207, 261]
        self.max_sizes = [45, 99, 153, 207, 261, 315]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            # Creates Cartiesian product
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # centre point of default boxes
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k
                # Small square of size (s_k, s_k)
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]
                # Large square of size (sqrt(s_k * s_(k+1)), sqrt(s_k * s_(k+1)))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]
                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
            output = torch.tensor(mean, device=self.device).view(-1, 4)
            output.clamp_(max=1, min=0)
        return output
