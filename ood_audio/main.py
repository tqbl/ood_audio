import json
import pickle
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats

import cli


def main(args):
    """Execute a task based on the given command-line arguments.

    This function is the main entry-point of the program. It allows the
    user to extract features, train a model, compute predictions, and
    evaluate predictions using the command-line interface.
    """
    from datasets import Datasets

    datasets = Datasets(args.dataset_path)

    if args.command == 'extract':
        extract(datasets.get(args.dataset), args)
    elif args.command == 'train':
        train(datasets.get('training'), args)
    elif args.command == 'predict':
        predict(datasets.get(args.dataset), args)
    elif args.command == 'evaluate':
        if isinstance(args.training_id, list):
            evaluate_all(datasets.get('test'), args)
        else:
            evaluate(datasets.get('test'), args)


def extract(dataset, args):
    """Extract feature vectors from the given dataset.

    Args:
        dataset (Dataset): Information about the dataset.
        args: Named tuple of configuration arguments.
    """
    import features

    # Use a logmel representation for feature extraction
    extractor = features.LogmelExtractor(args.sample_rate,
                                         args.n_fft,
                                         args.hop_length,
                                         args.n_mels,
                                         )

    # Ensure output directory exists and set file path
    os.makedirs(args.extraction_path, exist_ok=True)
    output_path = os.path.join(args.extraction_path, dataset.name + '.h5')

    # Log the values of the hyperparameters
    _log_parameters(os.path.join(args.extraction_path, 'parameters.json'),
                    sample_rate=args.sample_rate,
                    n_fft=args.n_fft,
                    hop_length=args.hop_length,
                    n_mels=args.n_mels,
                    )

    df = pd.read_csv(dataset.metadata_path, index_col=0)
    features.extract_dataset(dataset_path=dataset.path,
                             file_names=df.index,
                             extractor=extractor,
                             output_path=output_path,
                             recompute=args.recompute,
                             )


def train(dataset, args):
    """Train a model on a dataset.

    Args:
        dataset (Dataset): Information about the dataset.
        args: Named tuple of configuration arguments.
    """
    import pytorch.training as training
    import relabel
    import utils

    # Load training data and metadata
    x_train, df_train = _load_features(
        dataset, args.extraction_path, args.block_size)

    # Use subset of training set as validation set
    mask = df_train.validation == 1
    x_val = x_train[mask]
    df_val = df_train[mask]
    x_train = x_train[~mask]
    df_train = df_train[~mask]

    # Mask out training data based on user specification
    if args.mask:
        x_train, df_train = _mask_data(x_train, df_train, args.mask)

    # Encode labels as one-hot vectors
    y_train = pd.get_dummies(df_train.label)
    y_val = pd.get_dummies(df_val.label)

    # Relabel examples if relabeling is enabled
    if args.relabel:
        mask = df_train.manually_verified == 0
        y_pred = pd.read_csv(args.pseudolabel_path, index_col=0)
        y_conf = pd.read_csv(args.confidence_path, index_col=0).max(axis=1)
        y_train[mask] = relabel.relabel(y_train[mask], y_pred, y_conf,
                                        args.relabel_threshold,
                                        args.relabel_weight)

    # Ensure output directories exist
    model_path = os.path.join(args.model_path, args.training_id)
    log_path = os.path.join(args.log_path, args.training_id)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Save hyperparameters to disk
    params = {k: v for k, v in vars(args).items()
              if k in ['model',
                       'mask',
                       'seed',
                       'batch_size',
                       'n_epochs',
                       'lr',
                       'lr_decay',
                       'lr_decay_rate',
                       'augment',
                       'overwrite',
                       'relabel',
                       'relabel_threshold',
                       'relabel_weight',
                       ]}
    _log_parameters(os.path.join(model_path, 'parameters.json'), **params)

    # Standardize data and save statistics to disk
    mean, std = utils.statistics(x_train)
    x_train, x_val = utils.standardize([x_train, x_val], mean, std)
    pickle.dump((mean, std), open(os.path.join(model_path, 'scaler.p'), 'wb'))

    training.train(x_train, y_train.values,
                   x_val, y_val.values, df_val.index,
                   log_path, model_path, **params)


def predict(dataset, args):
    """Compute predictions on a dataset.

    This function uses an ensemble of trained models to compute the
    predictions, with the ensemble function being the arithmetic mean.
    The computed predictions are then saved to disk.

    Args:
        dataset (Dataset): Information about the dataset.
        args: Named tuple of configuration arguments.
    """
    import pytorch.odin as odin
    import pytorch.training as training
    import utils

    # Load and standardize data
    x, df = _load_features(dataset, args.extraction_path, args.block_size)
    model_path = os.path.join(args.model_path, args.training_id)
    mean, std = pickle.load(open(os.path.join(model_path, 'scaler.p'), 'rb'))
    x = utils.standardize(x, mean, std)

    # Use ODIN algorithm if enabled
    output_name = dataset.name
    predict_cb = None
    if args.odin:
        output_name += '_odin'
        predict_cb = odin.predict

    # Compute predictions for each model and ensemble using mean
    log_path = os.path.join(args.log_path, args.training_id)
    epochs = _determine_epochs(args.epochs, log_path)
    preds = [utils.timeit(lambda: training.predict(x, df, epoch, model_path,
                                                   callback=predict_cb),
                          f'[Epoch {epoch}] Computed predictions')
             for epoch in epochs]

    pred_mean = pd.concat(preds).groupby(level=0).mean()

    # Ensure output directory exists
    prediction_path = os.path.join(args.prediction_path, args.training_id)
    os.makedirs(prediction_path, exist_ok=True)

    # Save hyperparameters to disk
    _log_parameters(os.path.join(prediction_path, 'parameters.json'),
                    epochs=args.epochs,
                    )

    # Write predictions to disk
    pred_mean.to_csv(os.path.join(prediction_path, f'{output_name}.csv'))

    # Remove model files that were not used for prediction
    if args.clean:
        count = 0
        for path in Path(model_path).glob('model.[0-9][0-9].pth'):
            if int(str(path)[-6:-4]) not in epochs:
                path.unlink()
                count += 1
        print(f'Removed {count} unused model files')


def evaluate_all(dataset, args, verbose=True):
    """Run the evaluation for one or more models.

    If ``args.training_id`` is a list with multiple elements, each model
    corresponding to the training IDs is evaluated. Otherwise, this
    function is equivalent to :func:`evaluate`. If `verbose` is set to
    ``True``, the results are printed. When evaluating multiple models,
    this means that the scores are averaged and the mean and standard
    error of the mean are printed (for select metrics).

    Args:
        dataset (Dataset): Information about the dataset.
        args: Named tuple of configuration arguments.
        verbose (bool): Whether to print the results.

    Returns:
        list of pd.DataFrame: The results of the evaluation(s).
    """
    def _mean(scores):
        return np.mean(scores), scipy.stats.sem(scores)

    if len(args.training_id) == 1:
        args.training_id = args.training_id[0]
        return evaluate(dataset, args, verbose)

    scores = []
    for training_id in args.training_id:
        args.training_id = training_id
        scores.append(evaluate(dataset, args, verbose=False))

    for metric in ['AP', 'Recall']:
        for average in ['Macro Average', 'Micro Average']:
            mean, sem = _mean([s[metric][average] for s in scores])
            print(f'{metric} {average}: {mean:.3} \u00b1 {sem:.3}')

    return scores


def evaluate(dataset, args, verbose=True):
    """Evaluate the audio tagging predictions and record the results.

    Args:
        dataset (Dataset): Information about the dataset.
        args: Named tuple of configuration arguments.
        verbose (bool): Whether to print the results.

    Returns:
        pd.DataFrame: The results of the evaluation.
    """
    import evaluation

    # Load grouth truth data and predictions
    path = os.path.join(args.prediction_path,
                        args.training_id,
                        f'{dataset.name}.csv')
    y_pred = pd.read_csv(path, index_col=0)
    df_true = pd.read_csv(dataset.metadata_path, index_col=0)
    y_true = pd.get_dummies(df_true.loc[y_pred.index].label).values
    y_pred = y_pred.values

    # Evaluate audio tagging performance
    scores = evaluation.evaluate(y_true, y_pred)
    C = evaluation.confusion_matrix(y_true, y_pred)

    # Ensure output directory exist
    result_path = os.path.join(args.result_path, args.training_id)
    os.makedirs(result_path, exist_ok=True)

    # Write results to disk
    output_path = os.path.join(result_path, '%s_{}.csv' % dataset.name)
    scores.to_csv(output_path.format('scores'))
    C.to_csv(output_path.format('cmatrix'))

    # Print results (optional)
    if verbose:
        print('Confusion Matrix:\n', C.values, '\n')

        pd.options.display.float_format = '{:,.3f}'.format
        print(str(scores))

    return scores


def _load_features(dataset, data_path, block_size=128):
    """Load the features and the associated metadata for a dataset.

    The metadata is read from a CSV file and returned as a DataFrame.
    Each DataFrame entry corresponds to an instance in the dataset.

    Args:
        dataset (Dataset): Information about the dataset.
        data_path (str): Path to directory containing feature vectors.

    Returns:
        tuple: Tuple containing the array of feature vectors and the
        metadata of the dataset.
    """
    import features
    import utils

    # Load feature vectors from disk
    features_path = os.path.join(data_path, dataset.name + '.h5')
    x, n_blocks = utils.timeit(lambda: features.load_features(features_path,
                                                              block_size,
                                                              block_size // 4),
                               f'Loaded features of {dataset.name} dataset')
    # Reshape feature vectors: NxTxF -> NxTxFx1
    x = np.expand_dims(x, axis=-1)

    # Load metadata and duplicate entries based on number of blocks
    df = pd.read_csv(dataset.metadata_path, index_col=0)
    df = df.loc[np.repeat(df.index, n_blocks)]

    return x, df


def _mask_data(x, df, specs):
    """Mask data using the given specifications.

    Args:
        x (array_like): Array of data to mask.
        df (pd.DataFrame): Metadata used to apply the specifications.
        specs (dict): Specifications used to mask the data.
    """
    mask = np.ones(len(df), dtype=bool)
    for k, v in specs.items():
        if k[-1] == '!':
            mask &= df[k[:-1]] != v
        else:
            mask &= df[k] == v
    return x[mask], df[mask]


def _determine_epochs(spec, log_path):
    """Retrieve a list of epoch numbers based on the specification.

    The `spec` parameter may either be a list of epoch numbers or a
    tuple of the form ``(metric, n)``. In the former case, this function
    simply returns the list. In the latter case, this function will
    return the top `n` epochs based on the specified metric. This is
    determined using the training history file.

    Args:
        spec (list or tuple): List of epoch numbers or a tuple of the
            form ``(metric, n)``.
        log_path (str): Path to the log files directory.

    Returns:
        list: The relevant epoch numbers.
    """
    if type(spec) is list:
        return spec

    metric, n_epochs = spec
    df = pd.read_csv(os.path.join(log_path, 'history.csv'), index_col=0)
    df.sort_values(by=metric, ascending=metric in ['val_loss'], inplace=True)
    return df.index.values[:n_epochs]


def _log_parameters(output_path, **params):
    """Write the given parameter values to a JSON file.

    Args:
        output_path (str): Output file path.
        **params: Parameter values to log.
    """
    with open(output_path, 'w') as f:
        json.dump(params, f, indent=2)


if __name__ == '__main__':
    try:
        sys.exit(main(cli.parse_args()))
    except FileNotFoundError as error:
        sys.exit(error)
