def relabel(y_true, y_pred, y_conf, threshold=0.5, weight=0.5):
    """Use the relabeling algorithm to compute new labels.

    The algorithm first selects the examples that should be relabeled.
    This is determined using the predictions and confidence estimates of
    the auxiliary classifier, which are given by `y_pred` and `y_conf`,
    respectively. An example is relabeled if (1) the ground truth label
    does not agree with the predicted label and (2) the confidence
    estimate is above the specified threshold. The detected examples are
    relabeled as a convex combination of the ground truth label and the
    classifier prediction. The weight of the convex combination is given
    by the `weight` parameter.

    Args:
        y_true (pd.DataFrame): Ground truth labels to transform.
        y_pred (pd.DataFrame): Predictions of the classifier.
        y_conf (pd.DataFrame): Confidence estimates of the classifier.
        threshold (number): Threshold used to select OOD examples.
        weight (number): Weight used for relabeling.

    Returns:
        pd.DataFrame: The labels computed by the algorithm.
    """
    # Get clip-level ground truth labels
    y = y_true.groupby(level=0).first()

    # Ensure order matches that of ground truth
    y_pred = y_pred.loc[y.index]
    y_conf = y_conf.loc[y.index]

    # Determine which examples to relabel
    mask = (y.idxmax(axis=1) != y_pred.idxmax(axis=1)) & (y_conf > threshold)
    mask = y_true.index.isin(y[mask].index)

    y = y_true.copy()
    y[mask] = weight * y[mask] + (1 - weight) * y_pred.loc[y[mask].index]
    return y
