"""Sequence search: ISM-guided greedy (primary) and AdaLead evolutionary (ported, fixed).

Both operate on an editable slice of a full-context one-hot array and never touch the genomic
flanks, and both are seeded from the candidate's endogenous sequence (never a random start) —
sampling truly de-novo sequence puts the search off the training manifold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from regulonado.design.sequence import Seed, splice

__all__ = ["AdaLead", "AdaLeadConfig", "DesignState", "adalead", "ism_greedy"]


@dataclass(slots=True)
class DesignState:
    seed: Seed
    context: np.ndarray  # (4, context_length), from seed.window
    editable: slice  # == seed.editable, candidate span in context coordinates
    energy: float
    history: list[dict] = field(default_factory=list)


def _score(energy_fn, batch: np.ndarray) -> np.ndarray:
    """Run ``energy_fn`` and return a plain float64 array of per-sample energies.

    Accepts either an ``EnergyResult``-like object (``.energy``) or a bare tensor/array, so a
    toy objective can stand in for ``SpecificityEnergy`` in tests.
    """
    result = energy_fn(batch)
    energy = getattr(result, "energy", result)
    if hasattr(energy, "detach"):
        energy = energy.detach().cpu().numpy()
    return np.asarray(energy, dtype=np.float64)


# --------------------------------------------------------------------------- #
# ISM-guided greedy                                                          #
# --------------------------------------------------------------------------- #
def ism_greedy(
    energy_fn,
    seed: Seed,
    context: np.ndarray,
    *,
    rounds: int = 20,
    top_k: int = 1,
    positions: Sequence[int] | None = None,
    stride: int = 1,
    batch_size: int = 8,
    rng: np.random.Generator | None = None,
) -> DesignState:
    """Greedy in-silico mutagenesis: each round substitutes the ``top_k`` best non-conflicting

    single-base edits found by exhaustively scoring every alternative base at every (subsampled)
    editable position. Stops early once no substitution lowers the energy. ``positions``, when
    given, is an explicit set of context-coordinate positions to restrict the search to (e.g.
    resolved from a motif BED); otherwise every ``stride``-th editable position is considered.
    """
    del rng  # ISM is exhaustive at each position; no randomness to seed.

    editable = seed.editable
    if positions is not None:
        candidate_positions = [p for p in positions if editable.start <= p < editable.stop]
    else:
        candidate_positions = list(range(editable.start, editable.stop, stride))

    current = context.copy()
    baseline_energy = float(_score(energy_fn, current[None])[0])
    history: list[dict] = [{"round": 0, "energy": baseline_energy, "n_edits": 0}]
    state = DesignState(
        seed=seed, context=current, editable=editable, energy=baseline_energy, history=history
    )

    n_edits = 0
    for round_index in range(1, rounds + 1):
        proposals: list[tuple[int, int, np.ndarray]] = []
        for position in candidate_positions:
            column = current[:, position]
            current_base = int(column.argmax()) if column.any() else -1
            for base_index in range(4):
                if base_index == current_base:
                    continue
                mutated = current.copy()
                mutated[:, position] = 0
                mutated[base_index, position] = 1
                proposals.append((position, base_index, mutated))

        if not proposals:
            break

        energies = np.empty(len(proposals), dtype=np.float64)
        for batch_start in range(0, len(proposals), batch_size):
            chunk = proposals[batch_start : batch_start + batch_size]
            arrays = np.stack([p[2] for p in chunk])
            energies[batch_start : batch_start + len(chunk)] = _score(energy_fn, arrays)

        order = np.argsort(energies)
        accepted: list[tuple[int, int]] = []  # (position, base_index)
        used_positions: set[int] = set()
        for i in order:
            if energies[i] >= state.energy:
                break
            position, base_index, _ = proposals[i]
            if position in used_positions:
                continue
            accepted.append((position, base_index))
            used_positions.add(position)
            if len(accepted) >= top_k:
                break

        if not accepted:
            break

        for position, base_index in accepted:
            current[:, position] = 0
            current[base_index, position] = 1
            n_edits += 1

        new_energy = float(_score(energy_fn, current[None])[0])
        history.append(
            {
                "round": round_index,
                "energy": new_energy,
                "n_edits": n_edits,
                "positions": [position for position, _ in accepted],
            }
        )
        state.energy = new_energy

    state.context = current
    return state


# --------------------------------------------------------------------------- #
# AdaLead (ported from to_integrate/enhancer_toolkit/generators.py, fixed)    #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AdaLeadConfig:
    rounds: int = 20
    population_size: int = 20  # was hardcoded to 1 in the skeleton, disabling recombination
    mu: float = 1.0
    recomb_rate: float = 0.1
    threshold: float = 0.1
    rho: int = 2
    model_queries_per_batch: int | None = None
    max_attempts: int = 10_000
    energy_threshold: float = float("inf")


class AdaLead:
    """Adaptive evolutionary search, seeded from the candidate's endogenous insert."""

    def __init__(
        self,
        energy_fn,
        seed: Seed,
        context: np.ndarray,
        config: AdaLeadConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.energy_fn = energy_fn
        self.seed = seed
        self.context = context
        self.editable = seed.editable
        self.seq_len = self.editable.stop - self.editable.start
        self.config = config
        self.rng = rng or np.random.default_rng()
        self.model_cost = 0

    def get_fitness(self, inserts: Sequence[np.ndarray]) -> np.ndarray:
        for insert in inserts:
            assert insert.shape[1] == self.seq_len, "insert width must equal the editable width"
        self.model_cost += len(inserts)
        arrays = np.stack([splice(self.context, insert, self.editable.start) for insert in inserts])
        return -_score(self.energy_fn, arrays)

    def _mutate(self, insert: np.ndarray, mu_rate: float) -> np.ndarray:
        mutated = insert.copy()
        mask = self.rng.random(self.seq_len) < mu_rate
        n_mutated = int(mask.sum())
        if n_mutated:
            positions = np.nonzero(mask)[0]
            new_bases = self.rng.integers(0, 4, size=n_mutated)
            mutated[:, positions] = 0
            mutated[new_bases, positions] = 1
        return mutated

    def _recombine_population(
        self, population: list[np.ndarray], recomb_rate: float
    ) -> list[np.ndarray]:
        if len(population) <= 1:
            return list(population)
        order = self.rng.permutation(len(population))
        shuffled = [population[i] for i in order]
        recombined: list[np.ndarray] = []
        for idx in range(0, len(shuffled) - 1, 2):
            first, second = shuffled[idx], shuffled[idx + 1]
            switch_points = self.rng.random(self.seq_len) < recomb_rate
            switch = (np.cumsum(switch_points) % 2 == 1)[None, :]
            recombined.append(np.where(switch, second, first))
            recombined.append(np.where(switch, first, second))
        return recombined if recombined else list(population)

    def _propose_sequences(
        self,
        initial_inserts: Sequence[np.ndarray],
        *,
        mu: float,
        recomb_rate: float,
        threshold: float,
        rho: int,
        model_queries_per_batch: int,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        measured_keys = {insert.tobytes() for insert in initial_inserts}
        measured_fitness = self.get_fitness(initial_inserts)

        top_fitness = measured_fitness.max()
        cutoff = top_fitness * (1 - np.sign(top_fitness) * threshold)
        parent_indices = np.argwhere(measured_fitness >= cutoff).reshape(-1).tolist()
        parents = [initial_inserts[i] for i in parent_indices] or list(initial_inserts)

        population_size = self.config.population_size
        sequences: dict[bytes, tuple[np.ndarray, float]] = {}
        roots = [parents[i % len(parents)] for i in range(population_size)]

        self.model_cost = 0
        while self.model_cost < model_queries_per_batch:
            for _ in range(rho):
                roots = self._recombine_population(roots, recomb_rate)

            root_fitness = self.get_fitness(roots)
            nodes = list(enumerate(roots))

            while nodes and self.model_cost < model_queries_per_batch:
                candidates: list[tuple[int, np.ndarray, bytes]] = []
                for idx, root in nodes:
                    if self.model_cost >= model_queries_per_batch:
                        break
                    mutated = self._mutate(root, mu / self.seq_len)
                    key = mutated.tobytes()
                    if key in measured_keys or key in sequences:
                        continue
                    candidates.append((idx, mutated, key))

                if not candidates:
                    break

                children = [candidate[1] for candidate in candidates]
                fitness = self.get_fitness(children)

                improved_nodes: list[tuple[int, np.ndarray]] = []
                for (idx, child, key), fit in zip(candidates, fitness):
                    sequences[key] = (child, float(fit))
                    measured_keys.add(key)
                    if fit > root_fitness[idx]:
                        improved_nodes.append((idx, child))

                for idx, child in improved_nodes:
                    roots[idx] = child
                    root_fitness[idx] = sequences[child.tobytes()][1]

                nodes = improved_nodes

        if not sequences:
            raise ValueError(
                "No sequences generated. Increase model_queries_per_batch or reduce "
                "population_size."
            )

        items = list(sequences.values())
        fitness_array = np.array([fit for _, fit in items])
        top_indices = np.argsort(-fitness_array)[:population_size]
        return [items[i][0] for i in top_indices], fitness_array[top_indices]

    def generate(self) -> DesignState:
        cfg = self.config
        endogenous_insert = self.context[:, self.editable].copy()
        assert endogenous_insert.shape[1] == self.seq_len

        # Seeded from the endogenous sequence + mutants of it — never a random start, which is
        # exactly what the endogenous-start strategy exists to avoid.
        population = [endogenous_insert]
        while len(population) < cfg.population_size:
            population.append(self._mutate(endogenous_insert, cfg.mu / self.seq_len))
        population = population[: cfg.population_size]

        queries_per_batch = cfg.model_queries_per_batch or cfg.population_size * 10

        baseline_energy = float(_score(self.energy_fn, self.context[None])[0])
        history: list[dict] = [{"round": 0, "energy": baseline_energy, "n_edits": None}]

        inserts = population
        fitness = self.get_fitness(inserts)
        best_index = int(np.argmax(fitness))
        best_insert = inserts[best_index]
        best_energy = -float(fitness[best_index])

        for round_index in range(1, cfg.rounds + 1):
            inserts, fitness = self._propose_sequences(
                inserts,
                mu=cfg.mu,
                recomb_rate=cfg.recomb_rate,
                threshold=cfg.threshold,
                rho=cfg.rho,
                model_queries_per_batch=queries_per_batch,
            )
            round_best = int(np.argmax(fitness))
            round_energy = -float(fitness[round_best])
            if round_energy < best_energy:
                best_energy = round_energy
                best_insert = inserts[round_best]
            history.append({"round": round_index, "energy": round_energy, "n_edits": None})

        final_context = splice(self.context, best_insert, self.editable.start)
        return DesignState(
            seed=self.seed,
            context=final_context,
            editable=self.editable,
            energy=best_energy,
            history=history,
        )


def adalead(
    energy_fn,
    seed: Seed,
    context: np.ndarray,
    config: AdaLeadConfig,
    rng: np.random.Generator | None = None,
) -> DesignState:
    return AdaLead(energy_fn, seed, context, config, rng=rng).generate()
