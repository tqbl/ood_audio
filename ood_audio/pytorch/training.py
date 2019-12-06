import glob
import os.path

import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import inference
import pytorch.models as models
import pytorch.utils as utils
from pytorch.utils import ImageLoader
from pytorch.utils import Logger

from pytorch.specaug import SpecAugment


def train(x_train, y_train, x_val, y_val, index_val,
          log_path, model_path, **params):
    """Train a neural network classifier and compute predictions.

    Args:
        x_train (np.ndarray): Training set data.
        y_train (np.ndarray): Training set labels.
        x_val (np.ndarray): Validation set data.
        y_val (np.ndarray): Validation set labels.
        index_val (pd.Index): Validation set file names.
        log_path (str): Path to directory in which to record logs.
        model_path (str): Path to directory in which to save models.
        **params: Keyword arguments for the hyperparameters.
    """
    if params['seed'] >= 0:
        _ensure_reproducibility(params['seed'])

    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Instantiate neural network
    model = models.create_model(params['model'], y_train.shape[-1]).to(device)

    # Repeat data along channel dimension if applicable
    n_channels = next(model.parameters()).shape[1]
    if n_channels > 1:
        x_train = x_train.repeat(n_channels, axis=-1)
        x_val = x_val.repeat(n_channels, axis=-1)

    # Use CCE loss and Adam optimizer
    criterion = utils.cross_entropy
    optimizer = Adam(model.parameters(), lr=params['lr'])
    # Use scheduler to decay learning rate regularly
    scheduler = StepLR(
        optimizer,
        step_size=params['lr_decay_rate'],
        gamma=params['lr_decay'],
    )

    # Load training state from last checkpoint if applicable
    if not params['overwrite']:
        initial_epoch = _load_checkpoint(model, optimizer,
                                         scheduler, model_path)
        if initial_epoch >= params['n_epochs']:
            return
    else:
        initial_epoch = 0

    # Use helper classes to iterate over data in batches
    batch_size = params['batch_size']
    loader_train = ImageLoader(x_train, y_train, device, transform=None,
                               batch_size=batch_size, shuffle=True)
    loader_val = ImageLoader(x_val, y_val, device, batch_size=batch_size)

    # Instantiate Logger to record training/validation performance and
    # save checkpoint to disk after every epoch.
    model_path = os.path.join(model_path, 'model.{epoch:02d}.pth')
    logger = Logger(log_path, model_path, params['overwrite'])

    for epoch in range(initial_epoch, params['n_epochs']):
        # Enable data augmentation after 10 epochs
        if epoch == 10 and params['augment']:
            loader_train.dataset.transform = SpecAugment()

        # Train model using training set
        pbar = tqdm(loader_train)
        pbar.set_description(f'Epoch {epoch}')
        _train(pbar, model.train(), criterion, optimizer, logger)

        # Evaluate model using validation set
        _validate(loader_val, index_val, model.eval(), criterion, logger)

        # Invoke learning rate scheduler
        scheduler.step()

        # Log results and save model to disk
        logger.step(model, optimizer, scheduler)

    logger.close()


def predict(x, df, epoch, model_path, batch_size=128, odin=False):
    """Compute predictions using a saved model.

    The model that was saved after the specified epoch is used to
    compute predictions. After block-level predictions are computed,
    they are merged to give clip-level predictions.

    Args:
        x (np.ndarray): Array of input data.
        df (pd.DataFrame): Associated metadata.
        epoch (int): Epoch number of the model to load.
        model_path (str): Path to directory containing saved models.
        batch_size (int): Number of instances to predict per batch.
        odin (bool): Whether to use the ODIN algorithm.

    Returns:
        np.ndarray: The clip-level predictions.
    """
    # Determine which device (GPU or CPU) to use
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load model from disk
    model_path = os.path.join(model_path, f'model.{epoch:02d}.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model = models.create_model(*checkpoint['creation_args'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    # Repeat data along channel dimension if applicable
    n_channels = next(model.parameters()).shape[1]
    if n_channels > 1:
        x = x.repeat(n_channels, axis=-1)

    loader = ImageLoader(x, device=device, batch_size=batch_size)
    if odin:
        y_pred = _odin(model, loader)
    else:
        with torch.no_grad():
            y_pred = torch.cat([model(batch_x).softmax(dim=1).data
                                for batch_x, in loader])
    return inference.merge_predictions(y_pred.cpu().numpy(), df.index)


def _odin(model, loader, temperature=2.0, epsilon=8e-5):
    """Compute predictions using the ODIN algorithm [1]_.

    Args:
        model (torch.nn.Module): Model used to compute predictions.
        loader (torch.utils.data.DataLoader): Dataset to predict.
        temperature (number): Parameter used to scale the logits.
        epsilon (number): Amount of noise to add to the inputs.

    References:
        .. [1] S. Liang, Y. Li, and R. Srikant, “Enhancing the
               reliability ability of out-of-distribution image
               detection in neural networks,” in ICLR, 2018.
    """
    criterion = utils.cross_entropy

    y_preds = []
    for batch_x, in loader:
        batch_x = Variable(batch_x, requires_grad=True)
        batch_y = model(batch_x)

        labels = Variable(batch_y.softmax(dim=1))
        loss = criterion(batch_y / temperature, labels)
        loss.backward()

        gradient = 2 * torch.ge(batch_x.grad.data, other=0).float() - 1

        # Perturb inputs in the opposite direction of the gradient
        batch_x = torch.add(batch_x.data, epsilon, -gradient)
        # Compute predictions for perturbed inputs
        with torch.no_grad():
            batch_y = model(Variable(batch_x))

        y_preds.append((batch_y / temperature).softmax(dim=1).data)

    return torch.cat(y_preds)


def _train(data, model, criterion, optimizer, logger):
    """Train the model for a single epoch."""
    for batch_x, batch_y in data:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

        logger.log('loss', loss.item())


def _validate(loader, index, model, criterion, logger):
    """Validate the model using the given data."""
    # Compute block-level predictions
    with torch.no_grad():
        y_pred = torch.cat([model(batch_x).softmax(dim=1).data
                            for batch_x, _ in loader]).cpu().numpy()

    # Convert to clip-level predictions
    y_true = loader.dataset.y.numpy()
    y_true = inference.merge_predictions(y_true, index).values
    y_pred = inference.merge_predictions(y_pred, index).values
    y_pred_b = inference.binarize_predictions(y_pred)

    loss = criterion(torch.as_tensor(y_pred), torch.as_tensor(y_true))
    logger.log('val_loss', loss.item())

    acc = metrics.accuracy_score(y_true, y_pred_b)
    logger.log('val_acc', acc)

    ap = metrics.average_precision_score(y_true, y_pred, average='micro')
    logger.log('val_mAP', ap)


def _load_checkpoint(model, optimizer, scheduler, model_path):
    """Load the training state from the last checkpoint.

    This function searches the directory specified by `model_path` for
    checkpoints and selects the latest one.
    """
    # Check model directory for existing checkpoints
    paths = glob.glob(os.path.join(model_path, 'model.*.pth'))
    epoch = len(paths)
    if epoch == 0:
        return epoch

    # Load training state from last checkpoint
    checkpoint = torch.load(sorted(paths)[-1])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])

    if epoch != checkpoint['epoch'] + 1:
        # The epoch to resume from is determined by the number of saved
        # checkpoint files. If this number doesn't agree with the number
        # that is recorded internally, raise an error.
        raise RuntimeError('Epoch mismath')

    return epoch


def _ensure_reproducibility(seed):
    """Ensure training is deterministic by using a fixed random seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
