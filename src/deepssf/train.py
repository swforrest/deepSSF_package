"""Training: loss function, early stopping, and fitting loops.

Key objects
-----------
``negativeLogLikeLoss``
    Custom NLL loss for the joint habitat-movement output.
``EarlyStopping``
    Checkpoint-and-stop helper based on validation-loss improvement.
``load_checkpoint`` / ``load_head_weights``
    Read weights back, whole model or a single sub-network.
``params_from_checkpoint``
    Rebuild the ``ModelParams`` a checkpoint was trained with.
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
#: Since 0.3.1 a format-2 file written from a :class:`~deepssf.model.ConvJointModel`
#: also carries ``model_params``: the ``ModelParams`` dict the model was built
#: with, so :func:`params_from_checkpoint` can rebuild the architecture instead
#: of the user re-typing it.  The key is *optional* rather than a format bump —
#: an older reader ignores it, and a file written before 0.3.1 simply lacks it
#: (:func:`params_from_checkpoint` falls back to reading the layer shapes).
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

    The three returned numbers are related by

    .. code-block:: text

        total_loss = habitat_loss + movement_loss + logZ

    **What logZ is.**  ``logZ`` is step 2's normaliser,
    ``logsumexp(habitat + movement)`` over the window — the log of

    .. code-block:: text

        Z = sum_over_cells  p_habitat(cell) * p_movement(cell)

    Both surfaces arrive already normalised over the window, so ``Z`` is the
    *overlap* between them: the average habitat probability weighted by where
    the movement kernel says the animal could actually go,
    ``Z = E_movement[p_habitat]``.  It runs from 1 (the two surfaces agree
    perfectly) down towards 0 (they are concentrated in different places).  A
    flat habitat surface gives ``Z = 1 / (H*W)``, i.e. ``logZ = -log(H*W)`` —
    which is why an untrained model reports ``total ≈ movement_loss``, the
    uniform habitat term and the normaliser cancelling exactly.

    **Why it matters.**  ``logZ`` is the term that makes this a *step
    selection* likelihood rather than a plain habitat model: it divides out the
    habitat available within reach, so habitat is fitted against availability
    rather than against the landscape at large.

    Writing the habitat surface as ``p_habitat = softmax(f)`` over the window,
    the habitat gradient of the joint loss is

    .. code-block:: text

        d/dtheta [ -log p_habitat(y) + logZ ]  =  E_q[ df/dtheta ] - df(y)/dtheta

    where ``q = p_habitat * p_movement / Z`` is the model's own joint
    prediction (the ``E_p_habitat`` terms contributed by the two pieces cancel
    exactly).  Minimising ``habitat_loss`` on its own instead gives

    .. code-block:: text

        d/dtheta [ -log p_habitat(y) ]        =  E_p_habitat[ df/dtheta ] - df(y)/dtheta

    Both are the same used-versus-available contrast: raise ``f`` at the
    observed cell, lower it at cells drawn from an availability distribution.
    They differ *only* in that distribution — the joint loss draws availability
    from ``q``, concentrated on the cells actually within reach, while
    ``habitat_loss`` draws it from ``p_habitat`` spread over the whole window.
    On a trained feral-pig model those sets have effective sizes (``exp`` of
    their entropy) of about 109 and 5460 cells of a 75x75 window, and the
    available covariates sit roughly 2.3x closer to the used cell's under the
    first than the second.

    Note that neither is a *spatial* operation.  The habitat CNN has no
    positional input — the scalar covariates are broadcast to constant layers —
    so it is translation-equivariant and cannot treat "near the animal"
    differently from "at the window edge"; identical covariates always get an
    identical value.  What changes between the two criteria is purely which
    covariate values enter the available side of the contrast.

    The practical consequence is that ``total_loss`` is not monotone in
    ``habitat_loss``: filters fitted to discriminate among near-identical
    neighbours are not the filters that best separate the used cell from the
    whole window, so the two can move in opposite directions.  ``logZ`` falling
    while ``habitat_loss`` rises is a *symptom* of that sharpening — with one
    chosen cell pushed up and its many unchosen neighbours pushed down,
    ``E_movement[p_habitat]`` falls on average — not a loophole the model is
    exploiting.  It does mean "best joint model" and "best habitat surface" can
    be different epochs; see ``EarlyStopping(checkpoint_on=...)``.

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

    checkpoint_on:
        Which loss decides that an epoch is a new best worth saving.

        ``'total'`` (default) always uses the combined validation loss — the
        joint likelihood, and the right criterion for the finished model.

        ``'active'`` uses the combined loss only while *every* head is
        training, and during a single-head stage keys the checkpoint on that
        head's own loss instead.

        This matters because the combined loss is ``habitat + movement +
        logZ``, and ``logZ`` — the overlap between the habitat surface and the
        movement kernel — moves with the habitat weights.  The two losses are
        the same used-versus-available contrast measured against different
        availability sets: the combined loss against the cells within reach,
        ``habitat`` alone against the whole window (see
        :class:`negativeLogLikeLoss` for the gradients).  So in a habitat-only
        stage, where the movement term is pinned, the combined loss can keep
        falling — habitat discriminating better among near neighbours — while
        the density habitat gives the observed location gets *worse*.  A
        checkpoint is then written every epoch, ending on a habitat surface
        well past its best.

        Which criterion you want is a real choice, not a bug to be fixed.
        ``'total'`` selects the best joint step-selection model, with habitat
        fitted against availability — that is the classical iSSF estimand, and
        the right one if the habitat surface is going to be mapped as a
        *selection* surface with ``predict_habitat_landscape``.  ``'active'``
        answers a different, RSF-flavoured question: which habitat surface best
        predicts used locations against uniform availability across the window.
        Its main use is as a **diagnostic** — it tells you whether the habitat
        head is learning anything at all, and keeps the epoch where it had
        learned the most.

        Keep the magnitudes in view before tuning this.  On a feral-pig run the
        habitat head sat only ~0.04-0.15 nats below a uniform surface while
        movement was ~4.4 nats below it, so the whole best-epoch question moved
        a component worth a few percent of the model's predictive power.

        Changing criterion mid-run resets ``best_score``: a habitat-only loss
        and a combined loss are not on the same scale, so the first epoch of a
        new stage always saves under ``'active'``.
    head_paths:
        Optional ``{'habitat': path, 'movement': path}``.  When given, each
        head *also* gets its own checkpoint, written whenever that head's
        validation loss reaches a new low, independently of the stage schedule
        and of *checkpoint_on*.  Use this to keep the best habitat surface of
        the whole run even though a later joint stage moves it — read one back
        with :func:`load_head_weights`.

    Notes
    -----
    Every checkpoint holds the **whole** ``model.state_dict()`` — both heads,
    always.  The heads are not stored in separate files unless *head_paths*
    asks for it, and even then each file is a complete model; what differs is
    the epoch it was captured at.


    The checkpoint is a dict with ``deepssf_checkpoint_format``,
    ``deepssf_version``, ``val_loss`` and ``state_dict`` keys (see
    :data:`CHECKPOINT_FORMAT`), not a bare ``state_dict`` as in ≤ 0.2.3.  Read it
    back with :func:`load_checkpoint`, which verifies the format::

        load_checkpoint(path, model)                 # loads in place
        state = load_checkpoint(path)["state_dict"]   # or just the weights

    When the model carries a ``params`` attribute — every
    :class:`~deepssf.model.ConvJointModel` does — its ``ModelParams`` are
    written to the file as well, so the architecture can be rebuilt for
    prediction with :func:`params_from_checkpoint` instead of being re-entered
    by hand.
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
        checkpoint_on: str = "total",
        head_paths: dict[str, str] | None = None,
    ) -> None:
        if monitor not in ("total", "both"):
            raise ValueError("monitor must be 'total' or 'both'")
        if checkpoint_on not in ("total", "active"):
            raise ValueError("checkpoint_on must be 'total' or 'active'")
        if head_paths:
            unknown = set(head_paths) - set(self.HEADS)
            if unknown:
                raise ValueError(
                    f"unknown head(s) in head_paths: {sorted(unknown)}; "
                    f"expected a subset of {self.HEADS}"
                )

        self.patience    = patience
        self.verbose     = verbose
        self.delta       = delta
        self.path        = path
        self.trace_func  = trace_func
        self.monitor     = monitor
        self.checkpoint_on = checkpoint_on
        self.head_paths  = dict(head_paths) if head_paths else {}

        self.counter     = 0
        self.best_score  = None
        self.early_stop  = False
        self.val_loss_min = float("inf")

        # Per-head patience state, used when monitor='both'
        self._head_best: dict[str, float | None] = {h: None for h in self.HEADS}
        self._head_counter: dict[str, int] = {h: 0 for h in self.HEADS}

        # Which loss the current checkpoint's best_score refers to.  Scores from
        # two different criteria cannot be compared, so a change forces a reset.
        self._criterion_key: tuple[str, ...] | None = None
        # Best-ever per-head loss for head_paths.  Deliberately separate from
        # _head_best, which reset() clears at every stage boundary — these
        # follow the whole run.
        self._head_ckpt_best: dict[str, float | None] = {h: None for h in self.HEADS}

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
        components = {"habitat": val_habitat, "movement": val_movement}
        active = tuple(active)
        unknown = set(active) - set(self.HEADS)
        if unknown:
            raise ValueError(
                f"unknown component(s) in active: {sorted(unknown)}; "
                f"expected a subset of {self.HEADS}"
            )

        # --- Choose the criterion that decides "is this a new best?" ---------
        criterion, criterion_key = val_loss, ("total",)
        if (
            self.checkpoint_on == "active"
            and active
            and set(active) != set(self.HEADS)
        ):
            missing = [h for h in active if components[h] is None]
            if missing:
                raise ValueError(
                    f"checkpoint_on='active' requires {' and '.join(missing)} "
                    "validation loss; pass val_habitat= and val_movement= from "
                    "the epoch's validation pass."
                )
            criterion = sum(components[h] for h in active)
            criterion_key = active

        if criterion_key != self._criterion_key:
            # A habitat-only loss and a combined loss are on different scales,
            # so carrying best_score across the boundary would either freeze
            # checkpointing or save unconditionally. Start the new criterion fresh.
            self.best_score = None
            self.val_loss_min = float("inf")
            self._criterion_key = criterion_key

        # Negate loss so higher score = better (allows simple "did we improve?" check)
        score = -criterion
        improved = self.best_score is None or score >= self.best_score + self.delta

        if improved:
            # First epoch under this criterion, or a new best: save the checkpoint
            self.best_score = score
            self._save(
                self.path, val_loss, model,
                criterion=criterion, criterion_name="+".join(criterion_key),
            )

        # --- Per-head checkpoints, if asked for ------------------------------
        # Tracked over the whole run, so a later joint stage cannot overwrite
        # the epoch at which a head was individually at its best.
        for head, head_path in self.head_paths.items():
            head_loss = components[head]
            if head_loss is None:
                raise ValueError(
                    f"head_paths includes {head!r} but no val_{head} was passed; "
                    "pass val_habitat= and val_movement= from the validation pass."
                )
            best = self._head_ckpt_best[head]
            if best is None or head_loss <= best - self.delta:
                self._head_ckpt_best[head] = head_loss
                self._save(
                    head_path, val_loss, model,
                    criterion=head_loss, criterion_name=head,
                )

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
        missing = [h for h, v in components.items() if v is None]
        if missing:
            raise ValueError(
                f"monitor='both' requires {' and '.join(missing)} validation "
                "loss; pass val_habitat= and val_movement= from the epoch's "
                "validation pass."
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

    def _save(
        self,
        path: str,
        val_loss: float,
        model: nn.Module,
        *,
        criterion: float,
        criterion_name: str,
    ) -> None:
        """Write a checkpoint.

        ``val_loss`` is always the combined validation loss, so the field keeps
        the same meaning in every file.  ``criterion``/``criterion_name`` record
        which loss actually decided this was the best epoch — they differ from
        ``val_loss`` for a per-head file or under ``checkpoint_on='active'``.
        """
        is_main = path == self.path
        if self.verbose and is_main:
            self.trace_func(
                f"{criterion_name} validation loss decreased "
                f"({self.val_loss_min:.6f} → {criterion:.6f}). Saving model…"
            )
        elif self.verbose:
            self.trace_func(
                f"  new best {criterion_name} ({criterion:.6f}) → {path}"
            )
        # Deferred import: deepssf/__init__ imports this module, so importing it
        # at module scope would be circular.
        from deepssf import __version__

        payload = {
            "deepssf_checkpoint_format": CHECKPOINT_FORMAT,
            "deepssf_version": __version__,
            "val_loss": float(val_loss),
            "checkpoint_criterion": criterion_name,
            "checkpoint_score": float(criterion),
            "state_dict": model.state_dict(),
        }

        # Record the architecture alongside the weights, so the model can be
        # rebuilt for prediction without the hyper-parameters being re-entered
        # by hand.  getattr because EarlyStopping accepts any nn.Module, not
        # only ConvJointModel.
        params = getattr(model, "params", None)
        if hasattr(params, "to_dict"):
            payload["model_params"] = params.to_dict()

        torch.save(payload, path)
        if is_main:
            self.val_loss_min = criterion


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
        The checkpoint metadata, with a ``state_dict`` key — and, for a file
        written by deepssf ≥ 0.3.1, a ``model_params`` dict describing the
        architecture (:func:`params_from_checkpoint` turns it back into
        ``ModelParams``).  Legacy files are returned in the same shape, with
        ``deepssf_checkpoint_format`` set to 1.

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


def _architecture_from_state_dict(state: dict) -> dict:
    """Recover the shape-determining hyper-parameters from saved weights.

    A conv weight is ``[out_channels, in_channels, k, k]`` and a linear weight
    is ``[out_features, in_features]``, so every hyper-parameter that changes a
    tensor shape is readable from the file — which is what makes it possible to
    load a checkpoint written before ``model_params`` was recorded, and to
    verify one that has it.
    """
    def convs(prefix: str) -> list[str]:
        keys = [k for k in state if k.startswith(prefix) and k.endswith(".weight")]
        # 'conv_habitat.conv2d.4.weight' → 4.  Sorted numerically so keys[0] is
        # the layer that sees the input stack, whatever order torch listed them.
        return sorted(keys, key=lambda k: int(k.rsplit(".", 2)[-2]))

    hab, move, ffn = convs("conv_habitat."), convs("conv_movement."), convs("fcn_movement_all.")
    missing = [
        name for name, keys in
        (("conv_habitat", hab), ("conv_movement", move), ("fcn_movement_all", ffn))
        if not keys
    ]
    if missing:
        raise RuntimeError(
            f"Cannot read the architecture from this checkpoint: no weights for "
            f"{', '.join(missing)}. It holds "
            f"{sorted({k.split('.')[0] for k in state})} — is it a ConvJointModel?"
        )

    first_conv = state[hab[0]]
    return {
        # The first habitat conv sees the whole input stack: raster bands plus
        # the scalar covariates, which are broadcast to grids and concatenated.
        "input_channels":      int(first_conv.shape[1]),
        "output_channels":     int(first_conv.shape[0]),
        "kernel_size":         int(first_conv.shape[2]),
        "n_conv_layers_hab":   len(hab),
        "n_conv_layers_move":  len(move),
        "dense_dim_in_all":    int(state[ffn[0]].shape[1]),
        "dense_dim_hidden":    int(state[ffn[0]].shape[0]),
        "num_movement_params": int(state[ffn[-1]].shape[0]),
    }


#: Geometry that leaves no trace in the saved tensors — it changes spatial
#: *sizes*, not the shapes of stored weights — so it cannot be recovered from a
#: checkpoint written before ``model_params`` was recorded.  These are the
#: package defaults, and the only values the rest of the architecture is
#: coherent with: stride 1 with padding 1 keeps the habitat branch
#: translation-equivariant, which :func:`~deepssf.predict.predict_habitat_landscape`
#: depends on.
_UNRECORDED_GEOMETRY = {
    "stride": 1,
    "padding": 1,
    "kernel_size_mp": 2,
    "stride_mp": 2,
    "dropout": 0.0,     # inactive under model.eval() anyway
    "batch_size": 1,    # stored on ModelParams but never read by a layer
}


def params_from_checkpoint(
    path: str,
    *,
    image_dim: int | None = None,
    pixel_size: float | None = None,
    device: str | None = None,
    n_scalar_covariates: int | None = None,
    map_location="cpu",
    **overrides,
):
    """Rebuild the :class:`~deepssf.model.ModelParams` a checkpoint was trained with.

    Weights only load into a model built with the same architecture, so
    predicting from a saved model means reproducing its hyper-parameters
    exactly.  Re-typing them in a prediction script is the obvious way to get
    them subtly wrong — a habitat branch one layer too shallow fails loudly, but
    a mis-set ``pixel_size`` quietly rescales the movement kernel.

    Checkpoints written by deepssf ≥ 0.3.1 carry the params dict, so it is
    simply read back.  Older files do not, and the architecture is recovered
    from the layer shapes instead (see :func:`_architecture_from_state_dict`);
    *image_dim* and *pixel_size* leave no trace in the weights and must then be
    supplied.

    Either way the result is checked against the saved tensors, so a params dict
    that disagrees with the weights it was stored beside raises here rather than
    at ``load_state_dict``.

    Parameters
    ----------
    path:
        Checkpoint to read.
    image_dim:
        Spatial window size in pixels.  Overrides the stored value; required
        for a checkpoint that has none.  The habitat branch is fully
        convolutional so it runs at any size, but the movement branch's
        flattened size pins this down to a narrow range, which is checked.
    pixel_size:
        Metres per pixel.  Overrides the stored value; required for a
        checkpoint that has none.  Take it from the raster transform of the
        layers being predicted on rather than typing it in, since it is what
        the movement kernel is scaled against.
    device:
        Device for the rebuilt model.  Defaults to :func:`~deepssf.utils.get_device`
        for *this* machine — deliberately not the device recorded in the file,
        which is where the model happened to be *trained*.
    n_scalar_covariates:
        Number of scalar covariates (``len(SCALAR_COLS)``).  Only used for a
        checkpoint with no stored params, and only to fill
        ``dim_in_nonspatial_to_grid`` / ``dense_dim_in_nonspatial``, which no
        layer reads.  The count that *does* matter is folded into
        ``input_channels``, which is read off the weights.
    map_location:
        Passed to :func:`torch.load`.
    **overrides:
        Any other ``ModelParams`` field, forced to the given value.  Use for a
        checkpoint whose training used non-default conv/pool geometry.

    Returns
    -------
    ModelParams
        Ready to pass to :class:`~deepssf.model.ConvJointModel`::

            params = params_from_checkpoint(
                "best_model.pt", image_dim=WINDOW_SIZE, pixel_size=PIXEL_SIZE
            )
            model = ConvJointModel(params).to(params.device)
            load_checkpoint("best_model.pt", model)
            model.eval()

    Raises
    ------
    ValueError
        If *image_dim* or *pixel_size* is neither stored nor given, or if
        *image_dim* would not flatten to the size the movement MLP expects.
    RuntimeError
        If a stored params dict contradicts the saved weights, or the file does
        not hold a ``ConvJointModel``.
    """
    # Deferred import: deepssf/__init__ imports this module, so importing the
    # model at module scope would risk a circular import.
    from deepssf.model import ModelParams, flattened_conv_dim

    ckpt = load_checkpoint(path, map_location=map_location)
    from_weights = _architecture_from_state_dict(ckpt["state_dict"])
    stored = ckpt.get("model_params")

    if stored is None:
        params = {**_UNRECORDED_GEOMETRY, **from_weights}
        n_scalars = 0 if n_scalar_covariates is None else int(n_scalar_covariates)
        params["dim_in_nonspatial_to_grid"] = n_scalars
        params["dense_dim_in_nonspatial"] = n_scalars
    else:
        params = dict(stored)
        # The weights are the authority: a params dict can be edited, and the
        # two disagreeing means one of them describes a different model.
        conflicts = {
            key: (params[key], value)
            for key, value in from_weights.items()
            if key in params and params[key] != value
        }
        if conflicts:
            detail = "; ".join(
                f"{k}: params say {p!r}, weights say {w!r}"
                for k, (p, w) in sorted(conflicts.items())
            )
            raise RuntimeError(
                f"{path} stores hyper-parameters that contradict its own weights "
                f"({detail}). The file is inconsistent — retrain, or pass the "
                "correct values as keyword overrides."
            )
        params.update(from_weights)

    for name, value in (("image_dim", image_dim), ("pixel_size", pixel_size)):
        if value is not None:
            params[name] = value
        elif params.get(name) is None:
            raise ValueError(
                f"{name} is not recorded in {path} and was not given. It cannot "
                "be recovered from the weights: pass "
                f"{name}=... (the value the model was trained with)."
            )

    params["device"] = device if device is not None else get_device()
    params.update(overrides)

    unknown = set(params) - set(ModelParams.FIELDS)
    if unknown:
        raise ValueError(
            f"unknown ModelParams field(s): {sorted(unknown)}; expected a subset "
            f"of {list(ModelParams.FIELDS)}"
        )

    # The window is only loosely pinned by the weights — max-pooling
    # floor-divides, so a range of window sizes flattens to the same length —
    # but a window outside that range cannot be what this model was trained on.
    flat = flattened_conv_dim(
        image_dim=params["image_dim"],
        n_conv_layers_move=params["n_conv_layers_move"],
        output_channels=params["output_channels"],
        kernel_size=params["kernel_size"],
        stride=params["stride"],
        padding=params["padding"],
        kernel_size_mp=params["kernel_size_mp"],
        stride_mp=params["stride_mp"],
    )
    if flat != params["dense_dim_in_all"]:
        raise ValueError(
            f"image_dim={params['image_dim']} flattens to {flat} after "
            f"{params['n_conv_layers_move']}x conv+maxpool, but the movement MLP "
            f"in {path} expects {params['dense_dim_in_all']}. Set image_dim to "
            "the window size the model was trained with."
        )

    return ModelParams(params)


def _grads_are_finite(model: nn.Module) -> bool:
    """True if every populated gradient in *model* is free of NaN and Inf."""
    return all(
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    )


#: Which ``ConvJointModel`` sub-modules belong to which head.  Mirrors the
#: split :func:`set_trainable` freezes on, and is what lets one head's weights
#: be pulled out of a full checkpoint.
HEAD_MODULE_PREFIXES: dict[str, tuple[str, ...]] = {
    "habitat": ("conv_habitat.",),
    "movement": ("conv_movement.", "fcn_movement_all.", "movement_grid_output."),
}


def load_head_weights(
    path: str,
    model: nn.Module,
    head: str,
    *,
    map_location=None,
) -> int:
    """Load *only* one sub-network's weights from a checkpoint.

    Checkpoints always contain the whole model, so loading a per-head file with
    :func:`load_checkpoint` would bring the other head along with it — at
    whatever epoch that file happened to be written.  This copies across just
    the modules belonging to *head* and leaves the rest of *model* untouched.

    The usual pattern with ``EarlyStopping(head_paths=...)`` is to load the
    final joint model normally and then overwrite one head::

        load_checkpoint("best_model.pt", model)
        load_head_weights("best_model_habitat.pt", model, "habitat")

    Note that this is a deliberate mix-and-match: the two heads then come from
    different epochs, so the *joint* likelihood of the result is not what
    either checkpoint recorded.  It gives the best habitat surface, not the
    best joint model.

    Parameters
    ----------
    path:
        Checkpoint to read.
    model:
        Model to load into, modified in place.
    head:
        ``'habitat'`` or ``'movement'``.
    map_location:
        Passed to :func:`torch.load`.

    Returns
    -------
    int
        How many tensors were copied.
    """
    if head not in HEAD_MODULE_PREFIXES:
        raise ValueError(
            f"unknown head {head!r}; expected one of "
            f"{sorted(HEAD_MODULE_PREFIXES)}"
        )

    state = load_checkpoint(path, map_location=map_location)["state_dict"]
    prefixes = HEAD_MODULE_PREFIXES[head]
    subset = {k: v for k, v in state.items() if k.startswith(prefixes)}
    if not subset:
        raise RuntimeError(
            f"No {head} parameters found in {path}; the checkpoint holds "
            f"{sorted({k.split('.')[0] for k in state})}"
        )

    # strict=False because `subset` is by construction missing the other head;
    # unexpected keys would mean the checkpoint does not match this model.
    incompatible = model.load_state_dict(subset, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {path} has parameters this model does not: "
            f"{incompatible.unexpected_keys}"
        )
    return len(subset)


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
