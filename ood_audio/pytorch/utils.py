from collections import OrderedDict
import os.path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class ImageLoader(DataLoader):
    """A PyTorch DataLoader compatible with :class:`ImageDataset`.

    Args:
        x (np.ndarray): Array of 3D data.
        y (np.ndarray): Optional labels for the data.
        device (torch.device): Target device for the data. Data will be
            transferred to this device (e.g. the GPU) before the loader
            retrieves it. A value of ``None`` indicates that the data
            should not be moved.
        transform: Optional callback for transforming `x`.
        **args: Other arguments that are passed to the parent class.
    """

    def __init__(self, x, y=None, device=None, transform=None, **args):
        super().__init__(ImageDataset(x, y, transform), **args)

        self.device = device

    def __iter__(self):
        """Return an iterator for iterating over mini-batches.

        The mini-batches are mapped to the target device on-the-fly.
        """
        def _to(data):
            return [t.to(self.device) for t in data]

        return map(_to, super().__iter__())


class ImageDataset(Dataset):
    """A PyTorch Dataset for images or other 3D data.

    Args:
        x (np.ndarray): Array of 3D data.
        y (np.ndarray): Optional labels for the data.
        transform: Optional callback for transforming `x`.
    """

    def __init__(self, x, y=None, transform=None):
        self.x = torch.FloatTensor(x).permute(0, 3, 1, 2)
        self.y = torch.FloatTensor(y) if y is not None else None
        self.transform = transform

    def __getitem__(self, index):
        """Return the data at the given index.

        Transforms the data if applicable. The associated label is also
        returned if it exists.

        Args:
            index (int): Index of the data to retrieve.

        Returns:
            tuple: ``(x, y)``, where ``y`` may be ``None``.
        """
        x = self.x[index]
        if self.transform:
            x = self.transform(x)

        if self.y is None:
            return x,
        return x, self.y[index]

    def __len__(self):
        """Return the number of instances in the dataset."""
        return len(self.x)


class Logger:
    """A helper class for logging training results.

    Args:
        model (torch.nn.Module): PyTorch model used for training.
        log_path (str): Path to directory in which to record logs.
        model_path (str): Path to directory in which to save models.
    """

    def __init__(self, log_path, model_path):
        self.log_path = log_path
        self.model_path = model_path
        self.results = OrderedDict()
        self.results_df = pd.DataFrame()
        self.tb_writer = SummaryWriter(log_path)

    def log(self, key, value):
        """Log a value for the specified key.

        The key may be the name of a metric, for example, and the value
        may be the score achieved by the model for that metric.

        Args:
            key (str): Key for which value is logged.
            value (number): Value to log.
        """
        if self.results.get(key) is None:
            self.results[key] = []
        self.results[key].append(value)

    def step(self, model, optimizer):
        """Invoke the logger's 'save' operation.

        The values that have been logged (using :func:`log`) since the
        last invocation are saved to disk -- both in a CSV file and in a
        TensorBoard log file. If multiple values were logged for a key
        (e.g. a value per mini-batch), these values are averaged so that
        the mean value is saved instead. The saved values are also
        printed in the format

            key1: value1, key2: value2, ...

        A training checkpoint is also saved to disk. This includes the
        model weights, the state of the optimizer, and the states of the
        PRNGs used by PyTorch.

        Args:
            model (torch.nn.Module): PyTorch model to be saved.
            optimizer (torch.optim.Optimizer): Training optimizer.
        """
        # Write results to CSV file
        results = OrderedDict((k, np.mean(v)) for k, v in self.results.items())
        self.results_df = self.results_df.append(results, ignore_index=True)
        self.results_df.to_csv(os.path.join(self.log_path, 'history.csv'))

        # Write results to TensorBoard log file
        epoch = self.results_df.index[-1]
        for key, value in results.items():
            self.tb_writer.add_scalar(key, value, epoch)
        self.tb_writer.file_writer.flush()

        # Save model state to disk
        checkpoint = {'epoch': epoch,
                      'creation_args': model.creation_args,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'rng_state': torch.get_rng_state(),
                      'cuda_rng_state': torch.cuda.get_rng_state(),
                      }
        torch.save(checkpoint, self.model_path.format(epoch=epoch, **results))

        # Print results to stdout
        print(', '.join(['{}: {:.4f}'.format(k, v)
                         for k, v in results.items()]))

        self.results.clear()

    def close(self):
        """Release resources used by the logger."""
        self.tb_writer.close()


def cross_entropy(y_pred, y_true):
    """Compute the categorical cross-entropy.

    Args:
        y_pred (torch.Tensor): Array of predictions.
        y_true (torch.Tensor): Array of target values.

    Returns:
        number: The categorical cross-entropy.
    """
    return (-y_true * y_pred.log_softmax(dim=1)).sum(dim=1).mean()
