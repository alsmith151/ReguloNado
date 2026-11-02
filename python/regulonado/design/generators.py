"""Gradient sequence optimisation and whole-enhancer diffusion proposal ranking."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from regulonado.design.search import DesignState, _extract_energy, _round_metrics
from regulonado.genomics import decode, one_hot


def gradient_search(
    energy_fn,
    seed,
    context,
    *,
    rounds=100,
    learning_rate=0.1,
    temperature=1.0,
    rng=None,
    on_round=None,
):
    """Straight-through categorical optimisation; retain the best discrete sequence.

    This is a simple gradient baseline, not an implementation of Ledidi. Fixed
    context is copied exactly. Only sequence logits are optimised.
    """
    if learning_rate <= 0 or temperature <= 0:
        raise ValueError("learning_rate and temperature must be positive")
    ensemble = getattr(energy_fn, "ensemble", None)
    if ensemble is not None and ensemble.mode != "resident":
        raise ValueError("Gradient design requires fold_mode: resident")
    device = getattr(ensemble, "device", "cpu")
    base = torch.tensor(context, dtype=torch.float32, device=device)
    editable = seed.editable
    initial = base[:, editable]
    rng = rng if rng is not None else np.random.default_rng(0)
    logits = (
        (
            initial * 2
            + torch.tensor(rng.normal(0, 0.05, initial.shape), dtype=torch.float32, device=device)
        )
        .detach()
        .requires_grad_()
    )
    optimizer = torch.optim.Adam([logits], lr=learning_rate)
    result = energy_fn(context[None])
    best_result = result
    state = DesignState(seed, context.copy(), editable, float(_extract_energy(result)[0]))
    state.history.append(
        _round_metrics(result, round_index=0, n_edits=0, sequence=decode(context[:, editable]))
    )
    if on_round:
        on_round(state.history[-1])
    for index in range(1, rounds + 1):
        probabilities = (logits / temperature).softmax(0)
        hard = F.one_hot(probabilities.argmax(0), 4).T.float()
        straight_through = hard + (probabilities - probabilities.detach())
        candidate = torch.cat(
            [base[:, : editable.start], straight_through, base[:, editable.stop :]], dim=1
        )[None]
        optimizer.zero_grad(set_to_none=True)
        result = energy_fn(candidate)
        loss = result.energy if hasattr(result, "energy") else result
        if not loss.requires_grad:
            raise ValueError("Design predictor does not support gradients through sequence input")
        (gradient,) = torch.autograd.grad(loss.sum(), logits)
        logits.grad = gradient
        optimizer.step()
        # Evaluate the updated discrete sequence through the ordinary prediction path.
        discrete = context.copy()
        discrete[:, editable] = F.one_hot(logits.argmax(0), 4).T.detach().cpu().numpy()
        scored = energy_fn(discrete[None])
        energy = float(_extract_energy(scored)[0])
        if energy < state.energy:
            state.context, state.energy = discrete, energy
            best_result = scored
        n_edits = int(np.any(state.context[:, editable] != context[:, editable], axis=0).sum())
        entry = _round_metrics(
            best_result,
            round_index=index,
            n_edits=n_edits,
            sequence=decode(state.context[:, editable]),
        )
        state.history.append(entry)
        if on_round:
            on_round(entry)
    return state


def diffusion_search(
    energy_fn,
    seed,
    context,
    *,
    checkpoint,
    activities,
    count=32,
    steps=64,
    guidance=1.0,
    rng=None,
    on_round=None,
    device="auto",
):
    """Generate complete cores from masks, then rank with the existing objective."""
    from regulonado.design.diffusion import condition_vector, load_checkpoint, sample

    rng = rng if rng is not None else np.random.default_rng(0)
    model, metadata = load_checkpoint(checkpoint, device)
    values, observed = condition_vector(metadata, activities)
    sequences = sample(
        model,
        values,
        observed,
        length=seed.editable.stop - seed.editable.start,
        count=count,
        steps=steps,
        guidance=guidance,
        seed=int(rng.integers(0, 2**31)),
    )
    result = energy_fn(context[None])
    state = DesignState(seed, context.copy(), seed.editable, float(_extract_energy(result)[0]))
    entry = _round_metrics(
        result, round_index=0, n_edits=0, sequence=decode(context[:, seed.editable])
    )
    state.history.append(entry)
    if on_round:
        on_round(entry)
    # Always select a generated sequence, even if the endogenous reference scores better.
    best = float("inf")
    for index, sequence in enumerate(dict.fromkeys(sequences), 1):
        candidate = context.copy()
        candidate[:, seed.editable] = one_hot(sequence)
        result = energy_fn(candidate[None])
        energy = float(_extract_energy(result)[0])
        if energy < best:
            state.context, state.energy, best = candidate, energy, energy
        entry = _round_metrics(
            result,
            round_index=index,
            n_edits=int(np.any(candidate != context, axis=0).sum()),
            sequence=sequence,
            generated=True,
        )
        state.history.append(entry)
        if on_round:
            on_round(entry)
    return state


def diverse_sequences(sequences, *, count, min_distance=0):
    """Greedily retain ranked, unique sequences with a minimum Hamming distance."""
    if count < 1 or min_distance < 0:
        raise ValueError("count must be positive and min_distance nonnegative")
    selected = []
    for sequence in sequences:
        if sequence in selected:
            continue
        if all(
            len(sequence) != len(other)
            or sum(a != b for a, b in zip(sequence, other)) >= min_distance
            for other in selected
        ):
            selected.append(sequence)
        if len(selected) >= count:
            break
    return selected
