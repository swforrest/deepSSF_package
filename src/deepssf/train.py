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
``make_optimisers``
    Create dual Adam optimisers and ReduceLROnPlateau schedulers.
``fit``
    Full training loop: train, validate, schedule, checkpoint, snapshot.
"""

from __future__ import annotations

import os
import warnings

import torch
from torch import nn, optim

from deepssf.utils import get_device

#: Version of the on-disk checkpoint layout written by :class:`EarlyStopping`.
#:
#: * *format 1* (deepssf ≤ 0.2.3) — a bare ``state_dict``, with no metadata.
#: * *format 2* (deepssf ≥ 0.3.0) — a dict with ``deepssf_checkpoint_format``,
#:   ``deepssf_version``, ``val_loss`` and ``state_dict`` keys.
#:
#: The bump exists because 0.3.0 changed the meaning of movement parameters
#: 2, 5, 8 and 11 (mixture weights moved from ``softmax(exp(raw))`` to
#: ``log_softmax(raw)``).  Format-1 files still load into a 0.3.0 model without
#: a key mismatch, so :func:`load_checkpoint` uses this marker to refuse them
#: loudly instead of silently misinterpreting four parameters.
CHECKPOINT_FORMAT = 2

# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class negativeLogLikeLoss(nn.Module):
    """Negative log-likelihood loss for the deepSSF joint model output.

    The model produces a [B, H, W, 2] tensor where the last dimension holds
    the log-densities of the habitat and movement sub-networks.  This loss:

    1. Sums the two log-density channels to obtain a combined log-density.
    2. Re-normalises with the log-sum-exp trick.
    3. Indexes the log-density at the observed next-step pixel coordinates.
    4. Returns ``(total_loss, habitat_loss, movement_loss)``.

    Parameters
    ----------
    reduction:
        ``'mean'`` (default), ``'median'``, ``'sum'``, or ``'none'``.
    freeze_movement:
        If ``True``, only the habitat surface is used for the combined loss
        (movement parameters are effectively frozen during that pass).
    """

    def __init__(self, reduction: str = "mean", freeze_movement: bool = False) -> None:
        super().__init__()
        if reduction not in ("mean", "median", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'median', 'sum', or 'none'")
        self.reduction = reduction
        self.freeze_movement = freeze_movement

    def forward(
        self,
        predict: torch.Tensor,
        target: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        predict:
            Shape [B, H, W, 2] — log-densities from the joint model.
        target:
            ``(px2, py2)`` — 1-D integer tensors of length B giving the
            column (x) and row (y) pixel index of the next observed step
            within the local crop.

        Returns
        -------
        ``(total_loss, habitat_loss, movement_loss)`` — each scalar
        (mean/median/sum) or 1-D [B] tensor (none).
        """
        # Unpack the two log-probability surfaces from the joint model output
        hab_surface  = predict[:, :, :, 0]
        move_surface = predict[:, :, :, 1]

        # When freeze_movement=True only habitat drives the combined loss; the
        # movement sub-network receives no gradient on this pass.
        pred_prod = hab_surface if self.freeze_movement else hab_surface + move_surface

        # One check on the combined surface rather than three separate ones:
        # each `.any()` forces a device→host sync, which is costly per batch on
        # CUDA and MPS.  warnings.warn de-duplicates by default, so a run that
        # goes bad reports once instead of flooding the log; train_loop is
        # responsible for skipping the affected updates.
        if not torch.isfinite(pred_prod).all():
            which = []
            if not torch.isfinite(hab_surface).all():
                which.append("habitat")
            if not torch.isfinite(move_surface).all():
                which.append("movement")
            warnings.warn(
                "Non-finite values in the "
                f"{' and '.join(which) or 'combined'} probability surface. "
                "Training updates using this batch will be skipped. This usually "
                "means the movement parameters have diverged — consider passing "
                "grad_clip to fit(), or lowering the movement learning rate.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Re-normalise the combined log surface so it integrates to 1 in prob space
        pred_prod = pred_prod - torch.logsumexp(pred_prod, dim=(1, 2), keepdim=True)

        px2, py2 = target
        # batch_idx selects one row per sample; together with py2/px2 this indexes
        # the log-probability at the observed next-step pixel for each batch item.
        batch_idx = torch.arange(len(px2), device=predict.device)

        # NLL is the negative log-prob at the observed location (lower = better fit)
        nll      = -pred_prod[batch_idx, py2, px2]
        hab_loss = -hab_surface[batch_idx, py2, px2]
        mov_loss = -move_surface[batch_idx, py2, px2]

        if self.reduction == "mean":
            return torch.mean(nll), torch.mean(hab_loss), torch.mean(mov_loss)
        if self.reduction == "median":
            return torch.median(nll), torch.median(hab_loss), torch.median(mov_loss)
        if self.reduction == "sum":
            return torch.sum(nll), torch.sum(hab_loss), torch.sum(mov_loss)
        return nll, hab_loss, mov_loss


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

    Notes
    -----
    The checkpoint is a dict with ``deepssf_checkpoint_format``,
    ``deepssf_version``, ``val_loss`` and ``state_dict`` keys (see
    :data:`CHECKPOINT_FORMAT`), not a bare ``state_dict`` as in ≤ 0.2.3.  Read it
    back with :func:`load_checkpoint`, which verifies the format::

        load_checkpoint(path, model)                 # loads in place
        state = load_checkpoint(path)["state_dict"]   # or just the weights
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
        # Negate loss so higher score = better (allows simple "did we improve?" check)
        score = -val_loss

        if self.best_score is None:
            # First epoch — always save
            self.best_score = score
            self._save(val_loss, model)
        elif score < self.best_score + self.delta:
            # No meaningful improvement: increment patience counter
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # New best: save checkpoint and reset counter
            self.best_score = score
            self._save(val_loss, model)
            self.counter = 0

    def _save(self, val_loss: float, model: nn.Module) -> None:
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased "
                f"({self.val_loss_min:.6f} → {val_loss:.6f}). Saving model…"
            )
        # Deferred import: deepssf/__init__ imports this module, so importing it
        # at module scope would be circular.
        from deepssf import __version__

        torch.save(
            {
                "deepssf_checkpoint_format": CHECKPOINT_FORMAT,
                "deepssf_version": __version__,
                "val_loss": float(val_loss),
                "state_dict": model.state_dict(),
            },
            self.path,
        )
        self.val_loss_min = val_loss


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def load_checkpoint(
    path: str,
    model: nn.Module | None = None,
    *,
    map_location=None,
    allow_legacy: bool = False,
    strict: bool = True,
) -> dict:
    """Load a checkpoint written by :class:`EarlyStopping`, checking its format.

    Parameters
    ----------
    path:
        Checkpoint file to read.
    model:
        If given, the weights are loaded into this model in place.
    map_location:
        Passed to :func:`torch.load` (e.g. ``'cpu'`` to read a GPU checkpoint).
    allow_legacy:
        Accept a pre-0.3.0 checkpoint (a bare ``state_dict``).  Off by default
        because movement parameters 2, 5, 8 and 11 changed meaning in 0.3.0:
        the file will load without error but those four values are interpreted
        differently, so the movement kernel will be wrong.  Only enable this if
        the checkpoint predates 0.3.0 *and* you are not relying on the movement
        sub-network — otherwise retrain.
    strict:
        Passed to ``model.load_state_dict``.

    Returns
    -------
    dict
        The checkpoint metadata, with a ``state_dict`` key.  Legacy files are
        returned in the same shape, with ``deepssf_checkpoint_format`` set to 1.

    Raises
    ------
    RuntimeError
        If the file is a legacy checkpoint and *allow_legacy* is ``False``, or
        if it was written by a newer checkpoint format than this version knows.
    """
    obj = torch.load(path, map_location=map_location, weights_only=True)

    # Format 1: a bare state_dict, i.e. a flat mapping of names to tensors.
    if "deepssf_checkpoint_format" not in obj:
        if not allow_legacy:
            raise RuntimeError(
                f"{path} is a pre-0.3.0 deepssf checkpoint (no format marker). "
                "deepssf 0.3.0 changed the parameterisation of the movement "
                "mixture weights, so parameters 2, 5, 8 and 11 in this file mean "
                "something different to the current model and would be loaded "
                "silently. Retrain the model, or pass allow_legacy=True if you "
                "understand the consequences."
            )
        obj = {"deepssf_checkpoint_format": 1, "state_dict": obj}

    found = obj["deepssf_checkpoint_format"]
    if found > CHECKPOINT_FORMAT:
        raise RuntimeError(
            f"{path} uses checkpoint format {found}, but this deepssf "
            f"({CHECKPOINT_FORMAT}) can only read up to format "
            f"{CHECKPOINT_FORMAT}. Upgrade deepssf to load it."
        )

    if model is not None:
        model.load_state_dict(obj["state_dict"], strict=strict)

    return obj


def _grads_are_finite(model: nn.Module) -> bool:
    """True if every populated gradient in *model* is free of NaN and Inf."""
    return all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )


def train_loop(
    dataloader_train,
    model: nn.Module,
    loss_fn,
    optimisers: tuple,
    *,
    skip_epoch0_training: bool = False,
    batch_size: int = 32,
    grad_clip: float | None = None,
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
    grad_clip:
        If set, clip the global gradient norm to this value before each
        optimiser step (``torch.nn.utils.clip_grad_norm_``).  ``None``
        (default) leaves gradients unclipped.

    Returns
    -------
    epoch_loss : torch.Tensor
        Mean loss over all batches.

    Notes
    -----
    Batches whose loss is not finite are skipped: the optimiser is not stepped
    and the batch is excluded from the epoch mean.  Stepping on a NaN or Inf
    gradient writes NaN into every weight it touches, after which the run is
    unrecoverable, so skipping keeps a single bad batch from ending training.
    A count is reported at the end of the epoch.
    """
    device = get_device()
    optimiser_movement, optimiser_habitat = optimisers

    num_batches = len(dataloader_train)
    size        = len(dataloader_train.dataset)
    model.train()
    epoch_loss  = 0.0
    n_finite    = 0
    n_skipped   = 0

    for batch, (x1, x2, x3, y, _) in enumerate(dataloader_train):
        # Move batch to the active compute device (MPS / CUDA / CPU)
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y  = tuple(t.to(device) for t in y)

        with torch.set_grad_enabled(not skip_epoch0_training):
            outputs = model((x1, x2, x3))
            total_loss, _, _ = loss_fn(outputs, y)

        # A non-finite loss cannot produce a usable update; stepping on it would
        # write NaN into the weights permanently.  Skip the batch instead.
        if not torch.isfinite(total_loss):
            n_skipped += 1
            if n_skipped <= 5:
                print(
                    f"  [warning] non-finite loss ({total_loss.item()}) at batch "
                    f"{batch}; skipping update"
                )
            continue

        epoch_loss += total_loss.detach()
        n_finite   += 1

        if not skip_epoch0_training:
            if optimiser_movement is not None:
                optimiser_movement.zero_grad()
            if optimiser_habitat is not None:
                optimiser_habitat.zero_grad()

            total_loss.backward()

            # Gradients can be non-finite even when the loss is not (e.g. a
            # density that underflows to zero at the observed pixel), so check
            # after backward as well and drop the update if anything is bad.
            if not _grads_are_finite(model):
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  [warning] non-finite gradients at batch {batch}; "
                          "skipping update")
                if optimiser_movement is not None:
                    optimiser_movement.zero_grad()
                if optimiser_habitat is not None:
                    optimiser_habitat.zero_grad()
                continue

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # The two sub-networks share a single backward pass but are updated
            # by separate optimisers. To prevent cross-contamination:
            # 1. Stash habitat gradients and zero them so the movement optimiser
            #    only updates movement parameters.
            habitat_grads = []
            for param in model.conv_habitat.parameters():
                g = param.grad.clone() if param.grad is not None else None
                habitat_grads.append(g)
                param.grad = None

            if optimiser_movement is not None:
                optimiser_movement.step()

            # 2. Zero movement-FCN gradients, restore habitat gradients, then
            #    update only the habitat sub-network.
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

    # Average over the batches that actually contributed, so a few skipped
    # batches do not silently deflate the reported loss.
    epoch_loss /= max(n_finite, 1)
    tag = "observation-only " if skip_epoch0_training else ""
    print(f"\nAvg {tag}training loss: {epoch_loss:>15f}")
    if n_skipped:
        print(
            f"  [warning] {n_skipped}/{num_batches} batches skipped this epoch "
            "(non-finite loss or gradients)"
        )
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
            y  = tuple(t.to(device) for t in y)
            total_loss, _, _ = loss_fn(model((x1, x2, x3)), y)
            test_loss += total_loss.detach()

    test_loss /= num_batches
    torch.cuda.empty_cache()
    print(f"Avg test loss: {test_loss:>15f}\n")
    return test_loss


# ---------------------------------------------------------------------------
# Optimiser factory
# ---------------------------------------------------------------------------

def make_optimisers(
    model: nn.Module,
    lr_habitat: float = 1e-4,
    lr_movement: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.1,
) -> tuple[tuple, tuple]:
    """Create Adam optimisers and ReduceLROnPlateau schedulers for the joint model.

    Parameters
    ----------
    model:
        ConvJointModel instance.
    lr_habitat:
        Learning rate for the habitat CNN sub-network.
    lr_movement:
        Learning rate for the movement FCN sub-network.
    scheduler_patience:
        Epochs without improvement before reducing the learning rate.
    scheduler_factor:
        Multiplicative factor for learning-rate reduction.

    Returns
    -------
    optimisers : (optimiser_movement, optimiser_habitat)
    schedulers : (scheduler_movement, scheduler_habitat)
    """
    opt_movement = optim.Adam(
        model.fcn_movement_all.parameters(), lr=lr_movement
    )
    opt_habitat = optim.Adam(
        model.conv_habitat.parameters(), lr=lr_habitat
    )
    sched_movement = optim.lr_scheduler.ReduceLROnPlateau(
        opt_movement, patience=scheduler_patience, factor=scheduler_factor
    )
    sched_habitat = optim.lr_scheduler.ReduceLROnPlateau(
        opt_habitat, patience=scheduler_patience, factor=scheduler_factor
    )
    return (opt_movement, opt_habitat), (sched_movement, sched_habitat)


# ---------------------------------------------------------------------------
# Per-epoch snapshot helper
# ---------------------------------------------------------------------------

def _save_snapshot(
    model: nn.Module,
    image_trim_pixels: int,
    window_size: int,
    dl_val,
    snapshot_item: int,
    epoch: int,
    history: dict,
    snapshot_dir: str,
    device: str,
) -> None:
    """Save a 2×2 figure: loss curve + habitat / movement / step surfaces."""
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        sample = dl_val.dataset[snapshot_item]
    except (TypeError, IndexError):
        return

    x1, x2, x3, _y, _ = sample
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    x3 = x3.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        out = model((x1, x2, x3))

    hab_log = out[0, :, :, 0].cpu().numpy()
    move_log = out[0, :, :, 1].cpu().numpy()
    step_log = hab_log + move_log

    # Edge pixels within n_conv_layers of the border have seen padded (-1) values
    # in at least one conv receptive field, making their outputs less reliable.
    # Mask them out (set to NaN) so they don't distort the snapshot visualisation.
    edge_mask = np.zeros_like(hab_log, dtype=bool)

    edge_mask[:, :image_trim_pixels] = True
    edge_mask[:, window_size - image_trim_pixels:] = True
    edge_mask[:image_trim_pixels, :] = True
    edge_mask[window_size - image_trim_pixels:, :] = True

    # Apply mask
    hab_log_plot = hab_log.copy()
    move_log_plot = move_log.copy()
    step_log_plot = step_log.copy()

    hab_log_plot[edge_mask] = np.nan
    step_log_plot[edge_mask] = np.nan

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].plot(history["train_losses"], label="train")
    axes[0, 0].plot(history["val_losses"], label="val")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("NLL loss")
    axes[0, 0].legend()
    axes[0, 0].set_title("Training loss")

    axes[0, 1].imshow(hab_log_plot, origin="upper", cmap="viridis")
    # axes[0, 1].imshow(np.exp(hab_log_plot), origin="upper", cmap="viridis")
    axes[0, 1].set_title(f"Habitat - log (epoch {epoch + 1})")

    axes[1, 0].imshow(move_log_plot, origin="upper", cmap="viridis")
    # axes[1, 0].imshow(np.exp(move_log_plot), origin="upper", cmap="viridis")
    axes[1, 0].set_title(f"Movement - log (epoch {epoch + 1})")

    axes[1, 1].imshow(step_log_plot, origin="upper", cmap="viridis")
    # axes[1, 1].imshow(np.exp(step_log_plot), origin="upper", cmap="viridis")
    axes[1, 1].set_title(f"Next step - log (epoch {epoch + 1})")

    plt.tight_layout()
    path = os.path.join(snapshot_dir, f"epoch_{epoch + 1:03d}.png")
    fig.savefig(path, dpi=80)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def fit(
    model: nn.Module,
    image_trim_pixels: int,
    window_size: int,
    dl_train,
    dl_val,
    loss_fn,
    optimisers: tuple,
    schedulers: tuple | None = None,
    *,
    n_epochs: int = 10,
    early_stopping: EarlyStopping | None = None,
    snapshot_dir: str | None = None,
    snapshot_item: int = 0,
    grad_clip: float | None = None,
) -> dict[str, list[float]]:
    """Train for *n_epochs* with per-epoch validation, scheduling, and snapshots.

    Parameters
    ----------
    model:
        ConvJointModel to train.
    dl_train:
        Training DataLoader.
    dl_val:
        Validation DataLoader.
    loss_fn:
        Callable returning ``(total_loss, habitat_loss, movement_loss)``.
    optimisers:
        ``(optimiser_movement, optimiser_habitat)`` from :func:`make_optimisers`.
    schedulers:
        ``(sched_movement, sched_habitat)`` — both ``ReduceLROnPlateau``,
        stepped each epoch on the validation loss.  Pass ``None`` to skip.
    n_epochs:
        Maximum number of epochs.
    early_stopping:
        :class:`EarlyStopping` instance, or ``None`` to disable.
    snapshot_dir:
        Directory for per-epoch 2×2 PNG snapshots.  ``None`` disables saving.
    snapshot_item:
        Index into ``dl_val.dataset`` for the snapshot sample.
    grad_clip:
        Global gradient-norm clip applied before each optimiser step.  ``None``
        (default) leaves gradients unclipped; a value around ``1.0``–``5.0`` is
        a reasonable safeguard if the movement sub-network diverges.

    Returns
    -------
    history : dict[str, list[float]]
        Keys: ``train_losses``, ``val_losses``,
        ``val_habitat_losses``, ``val_movement_losses``.
    """
    device = get_device()
    sched_mov, sched_hab = (
        schedulers if schedulers is not None else (None, None)
    )

    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "val_habitat_losses": [],
        "val_movement_losses": [],
    }

    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")

        train_loss = train_loop(
            dl_train, model, loss_fn, optimisers, grad_clip=grad_clip
        )

        # Validation — track all three loss components
        model.eval()
        val_total = val_hab = val_mov = 0.0
        n_val = len(dl_val)

        with torch.no_grad():
            for x1, x2, x3, y, _ in dl_val:
                x1 = x1.to(device)
                x2 = x2.to(device)
                x3 = x3.to(device)
                y = tuple(t.to(device) for t in y)
                total, hab, mov = loss_fn(model((x1, x2, x3)), y)
                val_total += total.detach().item()
                val_hab += hab.detach().item()
                val_mov += mov.detach().item()

        val_total /= n_val
        val_hab /= n_val
        val_mov /= n_val
        print(
            f"Val loss: {val_total:.6f}"
            f"  (hab: {val_hab:.6f}, mov: {val_mov:.6f})"
        )

        history["train_losses"].append(float(train_loss))
        history["val_losses"].append(val_total)
        history["val_habitat_losses"].append(val_hab)
        history["val_movement_losses"].append(val_mov)

        if sched_mov is not None:
            sched_mov.step(val_total)
        if sched_hab is not None:
            sched_hab.step(val_total)

        if snapshot_dir is not None:
            _save_snapshot(
                model,  image_trim_pixels, window_size, 
                dl_val, snapshot_item, epoch,
                history, snapshot_dir, device,
            )

        if early_stopping is not None:
            early_stopping(val_total, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    return history