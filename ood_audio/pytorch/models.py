import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGG(nn.Module):
    """A PyTorch Module implementing a VGG-based neural network.

    The model contains 4 convolutional blocks (see :class:`ConvBlock`),
    so 8 convolutional layers in total. After the convolutional layers,
    the feature maps are average-pooled in the spatial dimensions. The
    final fully-connected layer follows.

    Args:
        n_classes (int): Number of target classes.
    """

    def __init__(self, n_classes):
        super(VGG, self).__init__()

        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
            ConvBlock(in_channels=128, out_channels=256),
            ConvBlock(in_channels=256, out_channels=512)
        )
        self.classifier = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):
        """Apply this module's forward pass."""
        x = self.conv_layers(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.shape[:2])
        return self.classifier(x)


class ConvBlock(nn.Module):
    """A PyTorch Module implementing a convolutional block.

    The block is comprised of two convolutional layers followed by batch
    normalization, a ReLU non-linearity, and then an optional
    max-pooling operation.

    Args:
        in_channels (int): Number of input channels (feature maps).
        out_channels (int): Number of output channels (feature maps).
        pool_size (tuple): Size of the max pooling kernel. A value of
            ``(1, 1)`` disables pooling.
        kernel_size (int): Size of the convolving kernel.
        **args: Keyword arguments to pass to :func:`torch.nn.Conv2d`.
    """

    def __init__(self, in_channels, out_channels,
                 pool_size=(2, 2), kernel_size=3, **args):
        super(ConvBlock, self).__init__()

        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size,
                               padding=padding, bias=False, **args)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size,
                               padding=padding, bias=False, **args)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool_size = pool_size

    def forward(self, x):
        """Apply this module's forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.pool_size != (1, 1):
            x = F.max_pool2d(x, self.pool_size)
        return x


def create_model(model_type, n_classes):
    """Instantiate the specified PyTorch model.

    Args:
        model_type (str): Name of the model. Either ``'vgg'`` or
            ``'densenet'`` (case-insensitive).
        n_classes (int): Number of target classes.

    Returns:
        nn.Module: An instance of the specified PyTorch model.
    """
    if model_type.lower() == 'vgg':
        model = VGG(n_classes)
    elif model_type.lower() == 'densenet':
        model = models.densenet201(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, n_classes)
    else:
        raise ValueError(f'Unrecognized model type: {model_type}')

    # Save the arguments that were passed to create the model
    model.creation_args = (model_type, n_classes)

    return model
