import numpy as np
import pandas as pd
import sklearn.metrics as metrics

import inference
import utils


def evaluate(y_true, y_pred, threshold=-1):
    """Evaluate audio tagging performance.

    Three types of scores are returned:

      * Class-wise
      * Macro-averaged
      * Micro-averaged

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of prediction probabilities.
        threshold (number): Threshold used to binarize predictions.

    Returns:
        pd.DataFrame: Table of evaluation results.
    """
    y_pred_b = inference.binarize_predictions(y_pred, threshold)

    class_scores = compute_scores(y_true, y_pred, y_pred_b).T
    macro_scores = np.mean(class_scores, axis=0, keepdims=True)
    micro_scores = compute_scores(y_true, y_pred, y_pred_b, average='micro')

    # Create DataFrame of evaluation results
    data = np.concatenate((class_scores, macro_scores, micro_scores[None, :]))
    index = utils.LABELS + ['Macro Average', 'Micro Average']
    columns = ['AP', 'F-score', 'Precision', 'Recall']
    return pd.DataFrame(data, pd.Index(index, name='Class'), columns)


def compute_scores(y_true, y_pred, y_pred_b, average=None):
    """Compute prediction scores using several performance metrics.

    The following metrics are used:

      * Average Precision
      * F-score
      * Precision
      * Recall

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of prediction probabilities.
        y_pred_b (np.ndarray): 2D array of binary predictions.
        average (str): The averaging method. Either ``'macro'``,
            ``'micro'``, or ``None``, where the latter is used to
            disable averaging.

    Returns:
        np.ndarray: Scores corresponding to the metrics used.
    """
    # Compute average precision
    ap = metrics.average_precision_score(y_true, y_pred, average=average)

    # Compute precision and recall scores
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred_b, average=average)

    return np.array([ap, fscore, precision, recall])


def confusion_matrix(y_true, y_pred):
    """Compute the confusion matrix for the given predictions.

    Args:
        y_true (np.ndarray): 2D array of ground truth values.
        y_pred (np.ndarray): 2D array of predictions.

    Returns:
        pd.DataFrame: The confusion matrix.
    """
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    C = metrics.confusion_matrix(y_true, y_pred)
    return pd.DataFrame(C, index=utils.LABELS, columns=utils.LABELS)
