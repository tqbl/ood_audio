import collections
import os.path


Dataset = collections.namedtuple('Dataset',
                                 ['name',
                                  'path',
                                  'metadata_path',
                                  ])
"""Data structure encapsulating information about a dataset."""


class Datasets:
    """Data structure encapsulating all of the datasets.

    Args:
        root_path (str): Path to directory containing datasets.

    Attributes:
        training_set (Dataset): Training set information.
        test_set (Dataset): Test set information.
    """

    def __init__(self, root_path):
        self.training_set = Dataset(
            name='training',
            path=os.path.join(root_path, 'FSDnoisy18k.audio_train'),
            metadata_path='metadata/training.csv',
        )

        self.test_set = Dataset(
            name='test',
            path=os.path.join(root_path, 'FSDnoisy18k.audio_test'),
            metadata_path='metadata/test.csv',
        )

    def get(self, name):
        """Return the Dataset instance corresponding to the given name.

        Args:
            name (str): Name of the dataset.

        Returns:
            Dataset: The corresponding Dataset instance.
        """
        if name == 'training':
            return self.training_set
        elif name == 'test':
            return self.test_set
        raise ValueError(f"Unknown dataset: '{name}'")
