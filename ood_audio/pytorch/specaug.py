import numpy as np
import torch


class SpecAugment:
    """An implementation of SpecAugment data augmentation [1]_.

    This implements time and frequency masking for time-frequency
    representations. Warping is not supported. Vertical masks are used
    to mask whole vertical sections within some time interval [T1, T2],
    where T1 and T2 are selected randomly. Similarly, horizontal masks
    are used to mask whole horizontal sections within some frequency
    interval [F1, F2].

    Args:
        T (int): Maximum width of a vertical mask.
        F (int): Maximum height of a horizontal mask.
        mT (int): Number of vertical masks.
        mF (int): Number of horizontal masks.

    References:
        .. [1] D. S. Park, W. Chan, Y. Zhang, C. Chiu, B. Zoph, E. D.
               Cubuk, and Q. V. Le, “SpecAugment: A simple data
               augmentation method for automatic speech recognition,” in
               Interspeech, 2019.
    """

    def __init__(self, T=8, F=8, mT=8, mF=2):
        self.T = T
        self.F = F
        self.mT = mT
        self.mF = mF

    def __call__(self, x):
        """Apply SpecAugment masking on the given data.

        Args:
            x (torch.Tensor): Input data to be masked.

        Returns:
            torch.Tensor: The transformed data after masking.
        """
        width, height = x.shape[-2:]
        mask = torch.ones_like(x, requires_grad=False)

        for _ in range(self.mT):
            t_delta = np.random.randint(low=0, high=self.T)
            t0 = np.random.randint(low=0, high=width - t_delta)
            mask[:, t0:t0 + t_delta, :] = 0

        for _ in range(self.mF):
            f_delta = np.random.randint(low=0, high=self.F)
            f0 = np.random.randint(low=0, high=height - f_delta)
            mask[:, :, f0:f0 + f_delta] = 0

        return x * mask
