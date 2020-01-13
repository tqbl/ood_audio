import torch

import pytorch.utils as utils


def predict(model, loader, temperature=1.5, epsilon=5e-4):
    """Compute predictions using the ODIN algorithm [1]_.

    Args:
        model (torch.nn.Module): Model used to compute predictions.
        loader (torch.utils.data.DataLoader): Dataset to predict.
        temperature (number): Parameter used to scale the logits.
        epsilon (number): Amount of noise to add to the inputs.

    Returns:
        torch.Tensor: The predictions of the model.

    References:
        .. [1] S. Liang, Y. Li, and R. Srikant, “Enhancing the
               reliability ability of out-of-distribution image
               detection in neural networks,” in ICLR, 2018.
    """
    y_preds = []
    for batch_x, in loader:
        batch_y = model(batch_x.requires_grad_())

        # Compute loss between prediction and target. The target in this
        # case is just the prediction without temperature scaling.
        target = batch_y.softmax(dim=1)
        loss = utils.cross_entropy(batch_y / temperature, target)
        loss.backward()

        # Perturb inputs in the opposite direction of the gradient
        batch_x = batch_x - epsilon * batch_x.grad.sign()
        # Compute predictions for perturbed inputs
        with torch.no_grad():
            batch_y = model(batch_x)

        y_preds.append((batch_y / temperature).softmax(dim=1).data)

    return torch.cat(y_preds)
