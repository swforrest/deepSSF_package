"""Training: loss function, early stopping, and fitting loops.

Key objects
-----------
``negativeLogLikeLoss``
    Custom NLL loss for the joint habitat-movement output.
``EarlyStopping``
    Checkpoint-and-stop helper based on validation-loss improvement.
``train_loop``
    One-epoch training pass with separate habitat/movement optimisers.
``test_loop``
    Evaluation pass (no gradients).
"""

from __future__ import annotations

import torch
from torch import nn

from deepssf.utils import get_device


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class negativeLogLikeLoss(nn.Module):
    """Negative log-likelihood loss for the deepSSF joint model output.

    The model produces a [B, H, W, 2] tensor where the last dimension holds
    the log-densities of the habitat and movement sub-networks.  This loss:

    1. Sums the two log-density channels to obtain a combined log-density.
    2. Re-normalises with the log-sum-exp trick.
    3. Evaluates the log-density at the observed next-step pixel (target).
    4. Returns the mean/sum/pointwise negative log-likelihood.

    Parameters
    ----------
    reduction:
        ``'mean'`` (default), ``'sum'``, or ``'none'``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        predict:
            Shape [B, H, W, 2] — log-densities from the joint model.
        target:
            Shape [B, H, W] — 1 at the observed next-step pixel, 0 elsewhere.

        Returns
        -------
        Scalar (mean/sum) or [B, H, W] tensor (none).
        """
        pred = predict[:, :, :, 0] + predict[:, :, :, 1]

        if torch.isnan(pred).any():
            raise ValueError("NaN detected in model predictions")

        pred = pred - torch.logsumexp(pred, dim=(1, 2), keepdim=True)
        nll  = -1 * (pred * target)

        if torch.isnan(nll).any():
            raise ValueError("NaN detected in NLL computation")

        if self.reduction == "mean":
            return torch.mean(nll)
        if self.reduction == "sum":
            return torch.sum(nll)
        return nll


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving and save the best checkpoint.

    Parameters
    ----------
    patience:
        Epochs to wait after the last improvement before stopping.
    verbose:
        Print a message each time the checkpoint is saved.
    delta:
        Minimum improvement to qualify as a new best.
    path:
        File path for the saved checkpoint.
    trace_func:
        Callable used for log messages (default: ``print``).
    """

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = "checkpoint.pt",
        trace_func=print,
    ) -> None:
        self.patience    = patience
        self.verbose     = verbose
        self.delta       = delta
        self.path        = path
        self.trace_func  = trace_func

        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model…"
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_loop(
    dataloader_train,
    model: nn.Module,
    loss_fn,
    optimisers: tuple,
    *,
    skip_epoch0_training: bool = False,
    batch_size: int = 32,
) -> torch.Tensor:
    """Run one training epoch.

    Parameters
    ----------
    dataloader_train:
        Yields ``(x1, x2, x3, y, raster_transform)`` batches, where
        *x1* is spatial, *x2* is scalar-to-grid, *x3* is bearing, and
        *y* is the target pixel coordinates.
    model:
        The deepSSF joint model.
    loss_fn:
        Callable returning ``(total_loss, habitat_loss, movement_loss)``.
    optimisers:
        ``(optimiser_movement, optimiser_habitat)`` — either may be ``None``
        to freeze that sub-network.
    skip_epoch0_training:
        If ``True``, run a forward pass but skip backward/update steps.
        Useful for inspecting untrained-model outputs.
    batch_size:
        Used only for progress reporting.

    Returns
    -------
    epoch_loss : torch.Tensor
        Mean loss over all batches.
    """
    device = get_device()
    optimiser_movement, optimiser_habitat = optimisers

    num_batches = len(dataloader_train)
    size        = len(dataloader_train.dataset)
    model.train()
    epoch_loss  = 0.0

    for batch, (x1, x2, x3, y, _) in enumerate(dataloader_train):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)

        with torch.set_grad_enabled(not skip_epoch0_training):
            outputs = model((x1, x2, x3))
            total_loss, _, _ = loss_fn(outputs, y)

        epoch_loss += total_loss.detach()

        if not skip_epoch0_training:
            if optimiser_movement is not None:
                optimiser_movement.zero_grad()
            if optimiser_habitat is not None:
                optimiser_habitat.zero_grad()

            total_loss.backward()

            # Save then zero habitat gradients so the movement optimiser
            # updates only movement parameters.
            habitat_grads = []
            for param in model.conv_habitat.parameters():
                habitat_grads.append(param.grad.clone() if param.grad is not None else None)
                param.grad = None

            if optimiser_movement is not None:
                optimiser_movement.step()

            # Zero movement-FCN gradients, restore habitat gradients, then
            # update the habitat sub-network.
            for param in model.fcn_movement_all.parameters():
                param.grad = None
            for i, param in enumerate(model.conv_habitat.parameters()):
                param.grad = habitat_grads[i]

            if optimiser_habitat is not None:
                optimiser_habitat.step()

        if batch % 10 == 0:
            current = batch * batch_size + len(x1)
            tag = "[obs only] " if skip_epoch0_training else ""
            print(f"{tag}loss: {total_loss.item():>15f}  [{current:>5d}/{size:>5d}]")

        torch.cuda.empty_cache()

    epoch_loss /= num_batches
    tag = "observation-only " if skip_epoch0_training else ""
    print(f"\nAvg {tag}training loss: {epoch_loss:>15f}")
    return epoch_loss


def test_loop(dataloader_test, model: nn.Module, loss_fn) -> torch.Tensor:
    """Evaluate the model on a held-out dataset (no gradients).

    Parameters
    ----------
    dataloader_test:
        Yields ``(x1, x2, x3, y, raster_transform)`` batches.
    model:
        The deepSSF joint model.
    loss_fn:
        Callable returning ``(total_loss, habitat_loss, movement_loss)``.

    Returns
    -------
    test_loss : torch.Tensor
        Mean loss over all batches.
    """
    device      = get_device()
    num_batches = len(dataloader_test)
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for x1, x2, x3, y, _ in dataloader_test:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            total_loss, _, _ = loss_fn(model((x1, x2, x3)), y)
            test_loss += total_loss.detach()

    test_loss /= num_batches
    torch.cuda.empty_cache()
    print(f"Avg test loss: {test_loss:>15f}\n")
    return test_loss