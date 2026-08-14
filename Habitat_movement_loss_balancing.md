# Balancing the habitat and movement components of the deepSSF loss

A design record for the training changes introduced after 0.3.0. It explains why
the habitat sub-network appeared not to learn, why the fix is not a loss weight,
and what the `stages` / `monitor="both"` machinery is for.

## The symptom

In a joint model trained on feral pig GPS data, the movement component of the
validation loss fell by roughly 1.2 nats over a run while the habitat component
moved by about 0.05 and sometimes drifted upward. On a 75×75 window a uniform
surface scores `log(75 × 75) = 8.635` nats; the habitat head finished at ≈ 8.25,
i.e. it had extracted about 0.4 nats of structure in total. Scoring the observed
next step on held-out data told the same story: mean habitat probability 0.000206
against a uniform 1/5625 = 0.000178, versus a mean movement probability of 0.054.

The natural reading is that the two components are summed into one loss and the
larger one drowns out the smaller, so the fix must be a weight on the habitat
term. That reading is wrong in an instructive way.

## What the loss actually is

`ConvJointModel` returns a `[B, H, W, 2]` stack in which **both** slices are
already log-normalised over the window — the habitat head subtracts its own
`logsumexp` (`Conv2d_block_spatial.forward`), and so does the movement head
(`Params_to_Grid_Block.forward`). `negativeLogLikeLoss` then sums them and
renormalises the sum:

```python
pred_prod = hab_surface + move_surface
pred_prod = pred_prod - torch.logsumexp(pred_prod, dim=(1, 2), keepdim=True)
nll = -pred_prod[batch_idx, py2, px2]
```

The `habitat_loss` and `movement_loss` it also returns are **diagnostics**:
`train_loop` and `test_loop` discard them (`total_loss, _, _ = loss_fn(...)`), and
only the epoch summary in `fit` reports them. There is no weighted sum anywhere,
and therefore no λ to turn.

Because both surfaces arrive normalised, the total decomposes exactly:

```
total = habitat_nll + movement_nll + logZ,    logZ = logsumexp(hab + mov) ≤ 0
```

So the *magnitudes* really are dominated by movement. But the gradient tells a
different story. Differentiating the total with respect to the habitat
parameters gives

```
∂total/∂θ_hab = -∇hab(y) + E_{p_combined}[∇hab]
```

— the standard conditional-logistic form: observed minus its expectation under
the combined surface. That is exactly the right step-selection gradient, with the
movement kernel playing the role of the availability distribution. **The habitat
head was never starved of gradient.** Adam normalises by gradient RMS on top of
that, so the raw scale difference between the components does not translate into
a proportional difference in step size either.

### Why weighting the habitat term would be wrong

The obvious intervention — minimise `total + λ · habitat_nll` — has a specific
failure mode. `habitat_nll` is the NLL of the habitat surface *on its own*, over
the whole window, and it is minimised by putting probability mass where animals
actually end up, which is overwhelmingly near the centre pixel. A habitat head
trained against it learns a distance-decay bump centred on the previous location:
it absorbs the movement structure instead of describing resource selection. The
same objection applies to `negativeLogLikeLoss(freeze_movement=True)`, which drops
the movement surface from the combined loss entirely — despite its name, it is not
a way to hold movement fixed while habitat trains. **Do not use it for that.**

## Where the asymmetry actually bites: model selection

The damage is downstream of the gradient. Before this change, both
`ReduceLROnPlateau` schedulers and `EarlyStopping` stepped on the combined
validation loss. Since that loss moves ~1.2 nats on movement and ~0.05 on
habitat, habitat's genuine improvement sits *inside* movement's epoch-to-epoch
noise band. The consequences:

- The run early-stops on movement's schedule. Habitat is still improving, but the
  combined loss has stopped moving by more than noise, so the patience counter
  climbs and training ends.
- The habitat learning rate is cut when the *movement* head plateaus.

Neither is fixed by changing the objective. Both are fixed by changing what is
monitored.

## Two bugs found alongside it

Neither is about loss balance, but no balancing scheme could be evaluated while
they stood.

**`conv_movement` belonged to no optimiser.** `make_optimisers` registered only
`fcn_movement_all` and `conv_habitat`. The movement CNN trunk received gradients
on every `backward()` but was never zeroed and never stepped, so its `.grad`
accumulated monotonically across the entire run. Two consequences: the trunk
stayed at its random initialisation forever, so the movement head was reading
fixed random features; and once the accumulating buffer overflowed,
`_grads_are_finite(model)` — which iterates over *every* parameter — began
returning `False`, at which point **every** subsequent update was silently
skipped, habitat's included, with only a `non-finite gradients at batch N`
warning to show for it.

**`skip_epoch0_training` disabled training.** The flag had no epoch awareness: it
was a per-call switch, and `fit` never passed it, so whatever its default was
applied to every epoch. With the default at `True`, the entire run executed
forward-only under `torch.set_grad_enabled(False)`. It also measured a
*training-set* baseline, when the number worth having is a validation one.

## The changes

### 1. Correct parameter ownership

`make_optimisers` now puts `conv_movement` and `fcn_movement_all` in the movement
optimiser as two param groups. `train_loop` zeroes gradients with
`model.zero_grad(set_to_none=True)` rather than per-optimiser, so no parameter can
fall outside the reset regardless of how the optimisers are configured.

The manual stash-and-restore of habitat gradients that used to bracket the two
optimiser steps has been deleted. The two optimisers own disjoint parameter sets
and `Adam.step()` only reads `.grad` for its own parameters, so clearing habitat
gradients before the movement step and putting them back afterwards was a no-op.

A regression test asserts that every parameter in `model.parameters()` is owned by
exactly one optimiser.

### 2. Per-head monitoring

Each `ReduceLROnPlateau` now steps on its own component of the validation loss,
and only while that component is training — stepping a frozen head's scheduler
would cut its learning rate for a lack of improvement it was never given the
chance to make.

`EarlyStopping` gains `monitor="both"`, which keeps a separate patience budget per
component and stops only once *every active* component has plateaued. Checkpointing
is deliberately unchanged: it still keys on the combined validation loss, because
the joint likelihood remains the correct model-selection criterion. Only the
stopping rule changes. `reset()` clears the counters at stage boundaries while
leaving the best score and saved checkpoint intact.

Note that with `delta=0.0` an exactly-unchanged loss counts as an improvement.
This is pre-existing behaviour, retained; real losses fluctuate, so a genuine
plateau still accumulates patience.

### 3. Staged (coordinate) training

`fit(stages=[...])` runs the sub-networks in turn:

```python
stages=[
    {"epochs": 10, "train": ("movement",)},
    {"epochs": 40, "train": ("habitat",)},
    {"epochs": 20, "train": ("habitat", "movement")},
]
```

Movement goes first so that a sensible availability kernel is in place before
habitat is fitted conditional on it. Habitat then runs to its own convergence on
its own early-stopping clock. A joint stage settles both.

A frozen branch is frozen through `set_trainable` (`requires_grad=False`) *and* by
withholding its optimiser. Crucially, **its surface stays in the likelihood as a
fixed offset** — only its weights stop moving. That is what makes this coordinate
ascent on the true joint objective rather than a reweighting of it, and it is the
difference between this mechanism and `freeze_movement=True`.

`reset_optimiser_state` (default `True`) clears the Adam moments of a branch when
it becomes active in a new stage, so a head frozen for forty epochs does not
resume on momentum pointing in a long-outdated direction.

Early stopping ends the current *stage* and advances to the next; for the final
stage that ends the run, so the behaviour is uniform.

### 4. A pre-training baseline

`skip_epoch0_training` is removed. `fit` instead runs one validation pass before
the first epoch and records it at index 0 of every history list, with
`train_losses[0] = nan` and `stage[0] = -1`. An untrained habitat surface is
uniform, so the baseline should land near `log(H × W)`; the gap between that and
the final `val_habitat_losses` is a direct, unambiguous read on how much the
habitat head actually learned. `history` also gains a `stage` key recording which
stage each epoch belonged to.

## Verifying it on a run

- The baseline line should print before epoch 1 with `hab ≈ log(H × W)`.
- No `non-finite gradients at batch N` warnings should appear — those were the
  signature of the accumulating `conv_movement` gradients.
- In a movement-only stage `val_habitat_losses` must be exactly flat, and in a
  habitat-only stage `val_movement_losses` must be exactly flat. If either moves,
  the freeze is not working.
- `val_habitat_losses` should fall materially below `log(H × W)` and flatten
  *before* the run ends, rather than the run ending first.

### 5. Correct covariate scaling

Found while investigating the flat habitat surface and fixed alongside it, since
it feeds the habitat CNN directly. `load_environmental_layers` divided by the
maximum rather than the range:

```python
data = (data - lo) / hi          # only correct when lo == 0
```

The failure is sign-dependent, which is why it went unnoticed:

| layer range | old result | correct |
|---|---|---|
| `lo=0, hi=100` | `[0, 1]` | `[0, 1]` — right by accident |
| `lo=-10000, hi=10000` (S2 indices ×10 000) | `[0, 2]` | `[0, 1]` |
| `lo=-100, hi=-10` (all negative) | `[0, -9]` — negative *and* axis-flipped | `[0, 1]` |
| `hi = 0` | `ZeroDivisionError` | `[0, 1]` |

The axis flip is the serious one: it reverses the sign of everything the habitat
CNN learns from that layer. The denominator is now `hi - lo`, with a constant
layer mapping to a flat 0 rather than dividing by zero.

Out-of-window padding stays at `-1.0`. With inputs correctly in `[0, 1]` that is
one full data range below the minimum, so padding remains cleanly distinguishable
from real values — which is what `image_trim_pixels` exists to keep out of the
snapshot visualisations.

**Checkpoints trained on the old scaling will not transfer.** The inputs have
changed; retrain.

## Still open

Found while investigating this, not addressed by it.

- **Habitat capacity.** Four conv layers at `output_channels=4` is ~27k parameters
  for the entire model, which may simply be too little to express much spatial
  structure. Worth revisiting now that the scaling is right and the training
  dynamics are trustworthy.
- **Per-band vs joint scaling.** A multi-band TIFF is scaled by its global min and
  max, so relative magnitudes between bands survive. That is right for bands
  sharing a natural range (NDVI/NDWI/NDMI) and wrong for bands in unrelated units.
