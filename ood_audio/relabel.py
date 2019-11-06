def relabel(y_true, y_pred, threshold=0.5, weight=0.5):
    """Use the relabeling algorithm to compute new labels.

    The algorithm first selects examples that are believed to be
    out-of-distribution (OOD). This is determined using the predictions
    of the auxiliary classifier, which are given by `y_pred`. An example
    is considered OOD if (1) the ground truth label does not agree with
    the predicted label and (2) the confidence of the classifier is
    above the specified threshold. After OOD examples are detected, they
    are relabeled as a convex combination of the ground truth label and
    the classifier prediction. The weight of the convex combination is
    given by the `weight` parameter.

    Args:
        y_true (pd.DataFrame): Ground truth labels to transform.
        y_pred (pd.DataFrame): Predictions of the classifier.
        threshold (number): Threshold used to select OOD examples.
        weight (number): Weight used for relabeling.

    Returns:
        pd.DataFrame: The labels computed by the algorithm.
    """
    y = y_true.groupby(level=0).first()
    y_pred = y_pred.loc[y.index]
    mask = (y.idxmax(axis=1) != y_pred.idxmax(axis=1)) \
        & (y_pred.max(axis=1) > threshold)
    mask = y_true.index.isin(y[mask].index)

    y = y_true.copy()
    y[mask] = weight * y[mask] + (1 - weight) * y_pred.loc[y[mask].index]
    return y
