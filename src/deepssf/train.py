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
    monitor:
        ``'total'`` (default) counts patience against the combined validation
        loss.  ``'both'`` counts patience separately for the habitat and
        movement components and stops only once *every* active component has
        plateaued.

        Use ``'both'`` for the joint model.  The combined loss is
        ``habitat + movement + logZ``, and because the movement surface is
        sharply peaked on the central pixels it improves by far more over a run
        than the diffuse habitat surface does — in a typical feral-pig run,
        ~1.2 nats against ~0.05.  Habitat's improvement therefore sits inside
        movement's epoch-to-epoch noise, and monitoring the total alone ends the
        run on movement's schedule while habitat is still learning.

    Notes
    -----
    The checkpoint is always keyed on the *combined* validation loss regardless
    of ``monitor``: the joint likelihood is the correct model-selection
    criterion.  ``monitor`` changes only when training is allowed to stop.


    The checkpoint is a dict with ``deepssf_checkpoint_format``,
    ``deepssf_version``, ``val_loss`` and ``state_dict`` keys (see
    :data:`CHECKPOINT_FORMAT`), not a bare ``state_dict`` as in ≤ 0.2.3.  Read it
    back with :func:`load_checkpoint`, which verifies the format::

        load_checkpoint(path, model)                 # loads in place
        state = load_checkpoint(path)["state_dict"]   # or just the weights
    """

    HEADS = ("habitat", "movement")

    def __init__(
        self,
        patience: int = 5,
        verbose: bool = False,
        delta: float = 0.0,
        path: str = "checkpoint.pt",
        trace_func=print,
        monitor: str = "total",
    ) -> None:
        if monitor not in ("total", "both"):
            raise ValueError("monitor must be 'total' or 'both'")

        self.patience    = patience
        self.verbose     = verbose
        self.delta       = delta
        self.path        = path
        self.trace_func  = trace_func
        self.monitor     = monitor

        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False
        self.val_loss_min = float("inf")

        # Per-head patience state, used when monitor='both'
        self._head_best: dict[str, float | None] = {h: None for h in self.HEADS}
        self._head_counter: dict[str, int] = {h: 0 for h in self.HEADS}

    def reset(self) -> None:
        """Clear the patience counters and the stop flag.

        Called at each stage boundary in staged training (see :func:`fit`) so a
        newly-unfrozen sub-network starts with a full patience budget.
        ``best_score`` and the saved checkpoint are deliberately left intact:
        the best model so far is still the best model.
        """
        self.counter     = 0
        self.early_stop  = False
        self._head_best   = {h: None for h in self.HEADS}
        self._head_counter = {h: 0 for h in self.HEADS}

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
        *,
        val_habitat: float | None = None,
        val_movement: float | None = None,
        active: tuple[str, ...] = HEADS,
    ) -> None:
        """Record an epoch's validation losses and update the stop decision.

        Parameters
        ----------
        val_loss:
            Combined validation loss.  Always drives checkpointing.
        model:
            Model to checkpoint when *val_loss* improves.
        val_habitat, val_movement:
            Per-component validation losses.  Required when ``monitor='both'``.
        active:
            Which components' patience gates stopping.  In staged training a
            frozen sub-network cannot improve, so counting its patience would
            end the stage for a head that was never given the chance to learn.
        """
        # Negate loss so higher score = better (allows simple "did we improve?" check)
        score = -val_loss
        improved = self.best_score is None or score >= self.best_score + self.delta

        if improved:
            # First epoch, or a new best: save the checkpoint
            self.best_score = score
            self._save(val_loss, model)

        if self.monitor == "total":
            self.counter = 0 if improved else self.counter + 1
            if not improved:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            self.early_stop = self.counter >= self.patience
            return

        # monitor == 'both': each component keeps its own patience budget, so a
        # plateaued movement head cannot end a run in which habitat is still
        # improving (and vice versa).
        components = {"habitat": val_habitat, "movement": val_movement}
        missing = [h for h, v in components.items() if v is None]
        if missing:
            raise ValueError(
                f"monitor='both' requires {' and '.join(missing)} validation "
                "loss; pass val_habitat= and val_movement= from the epoch's "
                "validation pass."
            )

        active = tuple(active)
        unknown = set(active) - set(self.HEADS)
        if unknown:
            raise ValueError(
                f"unknown component(s) in active: {sorted(unknown)}; "
                f"expected a subset of {self.HEADS}"
            )

        for head in self.HEADS:
            head_score = -components[head]
            best = self._head_best[head]
            if best is None or head_score >= best + self.delta:
                self._head_best[head] = head_score
                self._head_counter[head] = 0
            else:
                self._head_counter[head] += 1

        if not active:
            # Nothing is training, so nothing can plateau; leave the flag alone.
            self.early_stop = False
            return

        counters = ", ".join(
            f"{h}: {self._head_counter[h]}/{self.patience}" for h in active
        )
        self.trace_func(f"EarlyStopping counters — {counters}")
        self.early_stop = all(
            self._head_counter[h] >= self.patience for h in active
        )

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


def set_trainable(model: nn.Module, *, habitat: bool, movement: bool) -> None:
    """Toggle ``requires_grad`` on each sub-network of a ``ConvJointModel``.

    Used by staged training (see :func:`fit`) so that a frozen sub-network
    produces no gradients at all.  This is cheaper than discarding them after
    the fact, and means a frozen branch cannot silently accumulate gradients
    that no optimiser ever clears.

    Freezing a branch this way leaves its surface in the likelihood as a fixed
    offset — unlike ``negativeLogLikeLoss(freeze_movement=True)``, which drops
    the movement surface from the combined loss entirely.

    Parameters
    ----------
    model:
        ConvJointModel instance.
    habitat, movement:
        Whether the habitat and movement branches should receive gradients.
    """
    for param in model.conv_habitat.parameters():
        param.requires_grad_(habitat)
    for module in (model.conv_movement, model.fcn_movement_all):
        for param in module.parameters():
            param.requires_grad_(movement)


def train_loop(
    dataloader_train,
    model: nn.Module,
    loss_fn,
    optimisers: tuple,
    *,
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
        to freeze that sub-network.  Pair this with :func:`set_trainable` so
        the frozen branch does not compute gradients it will never use.
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

        # Zero on the model rather than per-optimiser: that covers every
        # parameter, including any branch frozen for this stage, so nothing can
        # be left out of the reset and quietly accumulate gradients.
        model.zero_grad(set_to_none=True)

        total_loss.backward()

        # Gradients can be non-finite even when the loss is not (e.g. a
        # density that underflows to zero at the observed pixel), so check
        # after backward as well and drop the update if anything is bad.
        if not _grads_are_finite(model):
            n_skipped += 1
            if n_skipped <= 5:
                print(f"  [warning] non-finite gradients at batch {batch}; "
                      "skipping update")
            model.zero_grad(set_to_none=True)
            continue

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # The two sub-networks share a single backward pass but own disjoint
        # parameter sets, and Adam only reads .grad for its own parameters —
        # so the two steps cannot contaminate each other and need no shuffling
        # of gradients between them.
        if optimiser_movement is not None:
            optimiser_movement.step()
        if optimiser_habitat is not None:
            optimiser_habitat.step()

        if batch % 10 == 0:
            current = batch * batch_size + len(x1)
            print(f"loss: {total_loss.item():>15f}  [{current:>5d}/{size:>5d}]")

        torch.cuda.empty_cache()

    # Average over the batches that actually contributed, so a few skipped
    # batches do not silently deflate the reported loss.
    epoch_loss /= max(n_finite, 1)
    print(f"\nAvg training loss: {epoch_loss:>15f}")
    if n_skipped:
        print(
            f"  [warning] {n_skipped}/{num_batches} batches skipped this epoch "
            "(non-finite loss or gradients)"
        )
    return epoch_loss


def _validate(
    model: nn.Module, dataloader, loss_fn, device
) -> tuple[float, float, float]:
    """Mean ``(total, habitat, movement)`` loss over *dataloader*, no gradients.

    The habitat and movement figures are diagnostics: only ``total`` is
    optimised.  Because both sub-network surfaces are already log-normalised,
    ``total = habitat + movement + logZ``, so the two components are directly
    comparable to the uniform baseline ``log(H * W)``.
    """
    model.eval()
    n_batches = len(dataloader)
    total_sum = hab_sum = mov_sum = 0.0

    with torch.no_grad():
        for x1, x2, x3, y, _ in dataloader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y  = tuple(t.to(device) for t in y)
            total, hab, mov = loss_fn(model((x1, x2, x3)), y)
            total_sum += total.detach().item()
            hab_sum   += hab.detach().item()
            mov_sum   += mov.detach().item()

    return total_sum / n_batches, hab_sum / n_batches, mov_sum / n_batches


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
    # The movement branch is conv_movement → fcn_movement_all; both must be
    # owned by the optimiser.  Before 0.3.1 conv_movement was in no optimiser at
    # all: it accumulated gradients across the whole run without ever being
    # zeroed or stepped, so the trunk stayed at its random initialisation and the
    # accumulating buffer eventually made _grads_are_finite fail for good.
    opt_movement = optim.Adam(
        [
            {"params": model.conv_movement.parameters()},
            {"params": model.fcn_movement_all.parameters()},
        ],
        lr=lr_movement,
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
# Per-epoch reporting helpers
# ---------------------------------------------------------------------------

def _lr_report(before: float, after: float) -> str:
    """Format a learning rate, noting a reduction made by the scheduler."""
    if after < before:
        return f"{after:.3e} (decreased from {before:.3e})"
    return f"{after:.3e}"


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


def _normalise_stages(
    stages: list[dict] | None, n_epochs: int
) -> list[tuple[int, tuple[str, ...]]]:
    """Validate *stages* and return it as ``[(epochs, active_components), ...]``.

    ``None`` becomes a single joint stage of *n_epochs*, i.e. the unstaged
    behaviour.
    """
    if stages is None:
        return [(n_epochs, EarlyStopping.HEADS)]

    if not stages:
        raise ValueError("stages must be a non-empty list, or None")

    normalised: list[tuple[int, tuple[str, ...]]] = []
    for i, stage in enumerate(stages):
        unknown_keys = set(stage) - {"epochs", "train"}
        if unknown_keys:
            raise ValueError(
                f"stage {i}: unknown key(s) {sorted(unknown_keys)}; "
                "expected 'epochs' and 'train'"
            )
        try:
            epochs = int(stage["epochs"])
            active = tuple(stage["train"])
        except KeyError as exc:
            raise ValueError(f"stage {i}: missing key {exc}") from None

        if epochs < 1:
            raise ValueError(f"stage {i}: epochs must be >= 1, got {epochs}")
        unknown = set(active) - set(EarlyStopping.HEADS)
        if unknown:
            raise ValueError(
                f"stage {i}: unknown component(s) in 'train': {sorted(unknown)}; "
                f"expected a subset of {EarlyStopping.HEADS}"
            )
        if not active:
            raise ValueError(f"stage {i}: 'train' must name at least one component")

        normalised.append((epochs, active))

    return normalised


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
    stages: list[dict] | None = None,
    reset_optimiser_state: bool = True,
    early_stopping: EarlyStopping | None = None,
    snapshot_dir: str | None = None,
    snapshot_item: int = 0,
    batch_size: int = 32,
    grad_clip: float | None = None,
) -> dict[str, list[float]]:
    """Train with per-epoch validation, scheduling, snapshots and optional staging.

    A validation pass runs *before* the first epoch and is recorded at index 0
    of every history list, giving an untrained baseline to measure against.  For
    a 75x75 window an untrained habitat surface should score close to
    ``log(75 * 75) = 8.635``, so the gap between that and the final
    ``val_habitat_losses`` is a direct read on how much habitat actually learned.

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
        ``(sched_movement, sched_habitat)`` — both ``ReduceLROnPlateau``.  Each
        is stepped on *its own* component of the validation loss, and only while
        that component is training.  Pass ``None`` to skip.
    n_epochs:
        Maximum number of epochs.  Ignored when *stages* is given.
    stages:
        Optional coordinate-ascent schedule, e.g.::

            stages=[
                {"epochs": 10, "train": ("movement",)},
                {"epochs": 40, "train": ("habitat",)},
                {"epochs": 20, "train": ("habitat", "movement")},
            ]

        Each stage names the sub-networks that receive updates; the others are
        frozen with :func:`set_trainable`, but their surfaces stay in the
        likelihood as a fixed offset.  This lets each component run to its own
        convergence without reweighting the objective.  Early stopping ends the
        current *stage* and advances to the next, so the last stage's early stop
        ends the run.  ``None`` (default) runs one joint stage of *n_epochs*.
    reset_optimiser_state:
        Clear the Adam moment estimates of a sub-network when it becomes active
        in a new stage, so it does not resume on momentum accumulated many
        epochs earlier.  Only applies when *stages* is given.
    early_stopping:
        :class:`EarlyStopping` instance, or ``None`` to disable.  Prefer
        ``monitor='both'`` here: the combined loss is dominated by the movement
        component, so stopping on it alone ends the run while habitat is still
        improving.
    snapshot_dir:
        Directory for per-epoch 2x2 PNG snapshots.  ``None`` disables saving.
    snapshot_item:
        Index into ``dl_val.dataset`` for the snapshot sample.
    grad_clip:
        Global gradient-norm clip applied before each optimiser step.  ``None``
        (default) leaves gradients unclipped; a value around ``1.0``–``5.0`` is
        a reasonable safeguard if the movement sub-network diverges.

    Returns
    -------
    history : dict[str, list[float]]
        Keys: ``train_losses``, ``val_losses``, ``val_habitat_losses``,
        ``val_movement_losses``, ``stage``.  **Index 0 is the pre-training
        baseline**: ``train_losses[0]`` is ``nan`` and ``stage[0]`` is ``-1``.
    """
    device = get_device()
    opt_mov, opt_hab = optimisers
    sched_mov, sched_hab = (
        schedulers if schedulers is not None else (None, None)
    )
    schedule = _normalise_stages(stages, n_epochs)

    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_losses": [],
        "val_losses": [],
        "val_habitat_losses": [],
        "val_movement_losses": [],
        "stage": [],
    }

    def record(train_loss: float, val: tuple[float, float, float], stage: int) -> None:
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val[0])
        history["val_habitat_losses"].append(val[1])
        history["val_movement_losses"].append(val[2])
        history["stage"].append(stage)

    # Untrained baseline, recorded at index 0 of every history list.
    baseline = _validate(model, dl_val, loss_fn, device)
    print(
        f"Baseline (untrained) val loss: {baseline[0]:.6f}"
        f"  (hab: {baseline[1]:.6f}, mov: {baseline[2]:.6f})"
    )
    record(float("nan"), baseline, -1)

    total_epochs = sum(epochs for epochs, _ in schedule)
    epoch = 0  # global counter, so snapshots stay ordered across stages

    for stage_idx, (stage_epochs, active) in enumerate(schedule):
        if stages is not None:
            print(
                f"\n=== Stage {stage_idx + 1}/{len(schedule)}: training "
                f"{' + '.join(active)} for up to {stage_epochs} epoch(s) ==="
            )
            set_trainable(
                model,
                habitat="habitat" in active,
                movement="movement" in active,
            )
            if reset_optimiser_state:
                # A sub-network frozen for many epochs would otherwise resume on
                # stale Adam moments pointing in a long-outdated direction.
                for name, opt in (("habitat", opt_hab), ("movement", opt_mov)):
                    if name in active and opt is not None:
                        opt.state.clear()
            if early_stopping is not None:
                early_stopping.reset()

        # Freeze by withholding the optimiser as well as requires_grad, so this
        # works whether or not set_trainable was applied.
        stage_optimisers = (
            opt_mov if "movement" in active else None,
            opt_hab if "habitat" in active else None,
        )

        for _ in range(stage_epochs):
            print(f"\nEpoch {epoch + 1}/{total_epochs}")

            train_loss = train_loop(
                dl_train, model, loss_fn, stage_optimisers,
                batch_size=batch_size, grad_clip=grad_clip,
            )

            val_total, val_hab, val_mov = _validate(model, dl_val, loss_fn, device)
            print(
                f"Val loss: {val_total:.6f}"
                f"  (hab: {val_hab:.6f}, mov: {val_mov:.6f})"
            )
            record(float(train_loss), (val_total, val_hab, val_mov), stage_idx)

            # Read the learning rates either side of the scheduler step so any
            # reduction can be flagged in the epoch summary.
            lr_mov_before = opt_mov.param_groups[0]["lr"]
            lr_hab_before = opt_hab.param_groups[0]["lr"]

            # Each scheduler steps on its own component, and only while that
            # component is training.  Stepping both on the combined loss let the
            # movement-dominated total decide when to cut habitat's learning
            # rate; stepping a frozen head's scheduler would cut its rate for a
            # lack of improvement it was never given the chance to make.
            if sched_mov is not None and "movement" in active:
                sched_mov.step(val_mov)
            if sched_hab is not None and "habitat" in active:
                sched_hab.step(val_hab)

            lr_mov = opt_mov.param_groups[0]["lr"]
            lr_hab = opt_hab.param_groups[0]["lr"]
            print(
                f"Learning rates: hab: {_lr_report(lr_hab_before, lr_hab)}"
                f", mov: {_lr_report(lr_mov_before, lr_mov)}"
            )

            if snapshot_dir is not None:
                _save_snapshot(
                    model,  image_trim_pixels, window_size,
                    dl_val, snapshot_item, epoch,
                    history, snapshot_dir, device,
                )

            epoch += 1

            if early_stopping is not None:
                early_stopping(
                    val_total, model,
                    val_habitat=val_hab, val_movement=val_mov,
                    active=active,
                )
                if early_stopping.early_stop:
                    if stages is None:
                        print("Early stopping triggered.")
                    else:
                        print(
                            f"Stage {stage_idx + 1} converged at epoch {epoch}; "
                            "moving on."
                        )
                    break

    return history
