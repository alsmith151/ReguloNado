"""Synthetic enhancer design: mutate candidates for cell-type specificity.

Optional stage — absent 'design:' config means these rules are never defined and 'rule all'
is unaffected. ISM is ~3 x window x folds forward passes per candidate and candidates are fully
independent, so the candidate BED is sharded across jobs/GPUs and merged back together.
"""

import re
import shlex
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path

DESIGN = config.get("design")
DESIGN_WANDB_RUN_ID = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
DESIGN_DIR = RESULTS / "design"


def _concat_files(paths, out_path):
    with open(out_path, "w") as out:
        for path in paths:
            out.write(Path(path).read_text())


def _merge_tsvs(paths, out_path):
    header = None
    with open(out_path, "w") as out:
        for path in paths:
            lines = Path(path).read_text().splitlines()
            if not lines:
                continue
            if header is None:
                header = lines[0]
                out.write(header + "\n")
            if len(lines) > 1:
                out.write("\n".join(lines[1:]) + "\n")


if DESIGN:

    def _design_run_names():
        explicit = DESIGN.get("design_runs")
        if explicit:
            return explicit
        holdout = DESIGN.get("holdout_run")
        return [name for name in RUN_NAMES if name != holdout]

    DESIGN_RUN_NAMES = _design_run_names()
    DESIGN_TARGETS = DESIGN["targets"]
    DESIGN_TARGET_NAMES = [target["name"] for target in DESIGN_TARGETS]
    DESIGN_TARGET_BY_NAME = {target["name"]: target for target in DESIGN_TARGETS}
    DESIGN_CHECKPOINT_DIRS = DESIGN.get("checkpoint_dirs")
    DESIGN_HOLDOUT_CHECKPOINT = DESIGN.get("holdout_checkpoint")
    if DESIGN_CHECKPOINT_DIRS and len(DESIGN_CHECKPOINT_DIRS) != len(DESIGN_RUN_NAMES):
        raise ValueError("design.checkpoint_dirs must match the number of design runs")

    _design_requested_shards = DESIGN.get("shards", 1)

    def _design_shard_names(wildcards):
        """Actual shard file stems, resolved only once shard_candidates has run.

        Candidates may themselves be an attribution-stage output
        (`results/attribution/{target}/core_regions.bed`), which does not exist
        at DAG-construction time. Reading its line count then — as a plain
        module-level computation — would either guess wrong (falling back to
        the requested shard count) or, worse, give a *different* answer on a
        second run once the file exists: a DAG whose job count depends on
        whether a previous run happened. `shard_candidates` is a checkpoint
        precisely so this resolves from its real output instead.
        """
        shard_dir = Path(checkpoints.shard_candidates.get().output[0])
        return sorted(path.stem for path in shard_dir.glob("*.bed"))

    def _design_run_dirs():
        return [str(phase_run_dir(run, PHASE_NAMES[-1])) for run in DESIGN_RUN_NAMES]

    def _design_holdout_dir():
        holdout = DESIGN.get("holdout_run")
        return str(phase_run_dir(holdout, PHASE_NAMES[-1])) if holdout else ""

    def _design_checkpoint_state_inputs(wildcards):
        if DESIGN_CHECKPOINT_DIRS:
            return list(DESIGN_CHECKPOINT_DIRS)
        return [
            str(phase_run_dir(run, PHASE_NAMES[-1]) / "trainer_state.json")
            for run in DESIGN_RUN_NAMES
        ]

    def _design_holdout_state_input(wildcards):
        if DESIGN_HOLDOUT_CHECKPOINT:
            return DESIGN_HOLDOUT_CHECKPOINT
        holdout = DESIGN.get("holdout_run")
        if not holdout:
            return []
        return str(phase_run_dir(holdout, PHASE_NAMES[-1]) / "trainer_state.json")

    def _design_flags(wildcards):
        """Merge common then per-target settings into repeatable, shell-safe CLI flags."""
        target = DESIGN_TARGET_BY_NAME[wildcards.target]
        merged = {}
        for settings in (DESIGN.get("common", {}), target.get("settings", {})):
            merged.update(_flatten_settings(settings))
        # Keep each configured design target in its own W&B project by default.
        # An explicit wandb_project in common/target settings remains authoritative.
        merged.setdefault("wandb_project", "regulonado-design")
        merged.setdefault("wandb_group", f"{wildcards.target}-{DESIGN_WANDB_RUN_ID}")
        flags = []
        for key, value in sorted(merged.items()):
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                flags.append(flag if value else "--no-" + key.replace("_", "-"))
            elif isinstance(value, (list, tuple)):
                for item in value:
                    flags.extend((flag, shlex.quote(str(item))))
            else:
                flags.extend((flag, shlex.quote(str(value))))
        return " ".join(flags)

    def _design_checkpoint_args():
        if DESIGN_CHECKPOINT_DIRS:
            return " ".join(f"--checkpoint {shlex.quote(str(path))}" for path in DESIGN_CHECKPOINT_DIRS)
        return ""

    def _design_holdout_args():
        if DESIGN_HOLDOUT_CHECKPOINT:
            return f"--holdout-checkpoint {shlex.quote(str(DESIGN_HOLDOUT_CHECKPOINT))}"
        return ""

    checkpoint shard_candidates:
        """Split candidates into shards, using the real row count of a file guaranteed to exist.

        A `checkpoint` (not a plain `rule`) because the shard count depends on
        this rule's own input content, which itself may only exist once an
        upstream rule (e.g. attribution's `merge_attributions`) has produced
        it — unknowable at DAG-construction time. Downstream rules resolve the
        actual shard count via `_design_shard_names()` instead of a
        precomputed module-level list.
        """
        input:
            DESIGN["candidates"],
        output:
            directory(str(DESIGN_DIR / "shards")),
        run:
            # Round-robin, not contiguous blocks, so chromosome clustering in the BED
            # doesn't make one shard dominate the runtime.
            lines = [line for line in Path(input[0]).read_text().splitlines() if line.strip()]
            n_shards = max(1, min(_design_requested_shards, len(lines))) if lines else (
                _design_requested_shards
            )
            Path(output[0]).mkdir(parents=True, exist_ok=True)
            buckets = [[] for _ in range(n_shards)]
            for index, line in enumerate(lines):
                buckets[index % n_shards].append(line)
            for shard_index, bucket in enumerate(buckets):
                text = "\n".join(bucket)
                (Path(output[0]) / f"{shard_index}.bed").write_text(text + "\n" if text else "")

    rule design_shard:
        input:
            shard=lambda w: str(Path(checkpoints.shard_candidates.get().output[0])
                                 / f"{w.shard}.bed"),
            intervals=config["inputs"]["intervals"],
            checkpoints=_design_checkpoint_state_inputs,
            holdout=_design_holdout_state_input,
        params:
            target=lambda w: DESIGN_TARGET_BY_NAME[w.target]["target"],
            group_by=lambda w: DESIGN_TARGET_BY_NAME[w.target].get("group_by", "source"),
            method=lambda w: DESIGN_TARGET_BY_NAME[w.target].get("method", "ism"),
            fasta=config["inputs"]["fasta"],
            dataset_dir=str(training_dataset_dir()),
            track_sheet=config["inputs"].get("track_sheet") or "",
            out_dir=lambda w: str(DESIGN_DIR / w.target / "shards" / w.shard),
            flags=_design_flags,
            resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
            design_run_dirs=_design_run_dirs(),
            holdout_run_dir=_design_holdout_dir(),
            explicit_checkpoint_args=_design_checkpoint_args(),
            explicit_holdout_args=_design_holdout_args(),
        output:
            tsv=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "designs.tsv"),
            fa=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "designs.fa"),
            bed=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "designs.bed"),
            trajectory=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "trajectory.tsv"),
            edits=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "edits.tsv"),
            run_json=str(DESIGN_DIR / "{target}" / "shards" / "{shard}" / "run.json"),
        resources:
            gpu=1,
            runtime=240,
        wildcard_constraints:
            target="|".join(re.escape(name) for name in DESIGN_TARGET_NAMES),
            shard=r"\d+",
        log:
            str(RESULTS / "logs" / "design_{target}_{shard}.log"),
        shell:
            r"""
            set -euo pipefail

            CHECKPOINT_ARGS=()
            if [ -n "{params.explicit_checkpoint_args}" ]; then
                CHECKPOINT_ARGS=({params.explicit_checkpoint_args})
            else
                for run_dir in {params.design_run_dirs}; do
                    CKPT=$(python {params.resolver:q} "$run_dir")
                    CHECKPOINT_ARGS+=(--checkpoint "$CKPT")
                done
            fi

            HOLDOUT_ARGS=()
            if [ -n "{params.explicit_holdout_args}" ]; then
                HOLDOUT_ARGS=({params.explicit_holdout_args})
            elif [ -n "{params.holdout_run_dir}" ]; then
                HCKPT=$(python {params.resolver:q} "{params.holdout_run_dir}")
                HOLDOUT_ARGS=(--holdout-checkpoint "$HCKPT")
            fi

            TRACK_SHEET_ARGS=()
            if [ -n "{params.track_sheet}" ]; then
                TRACK_SHEET_ARGS=(--track-sheet {params.track_sheet:q})
            fi

            regulonado design \
                --candidates {input.shard:q} \
                --intervals {input.intervals:q} \
                "${{CHECKPOINT_ARGS[@]}}" \
                "${{HOLDOUT_ARGS[@]}}" \
                --fasta {params.fasta:q} \
                --dataset-dir {params.dataset_dir:q} \
                "${{TRACK_SHEET_ARGS[@]}}" \
                --target {params.target:q} \
                --group-by {params.group_by:q} \
                --method {params.method:q} \
                {params.flags} \
                --out {params.out_dir:q} \
                > {log:q} 2>&1
            """

    rule merge_designs:
        input:
            tsv=lambda w: expand(
                str(DESIGN_DIR / w.target / "shards" / "{shard}" / "designs.tsv"),
                shard=_design_shard_names(w),
            ),
            fa=lambda w: expand(
                str(DESIGN_DIR / w.target / "shards" / "{shard}" / "designs.fa"),
                shard=_design_shard_names(w),
            ),
            bed=lambda w: expand(
                str(DESIGN_DIR / w.target / "shards" / "{shard}" / "designs.bed"),
                shard=_design_shard_names(w),
            ),
            trajectory=lambda w: expand(
                str(DESIGN_DIR / w.target / "shards" / "{shard}" / "trajectory.tsv"),
                shard=_design_shard_names(w),
            ),
            edits=lambda w: expand(
                str(DESIGN_DIR / w.target / "shards" / "{shard}" / "edits.tsv"),
                shard=_design_shard_names(w),
            ),
        output:
            tsv=str(DESIGN_DIR / "{target}" / "designs.tsv"),
            fa=str(DESIGN_DIR / "{target}" / "designs.fa"),
            bed=str(DESIGN_DIR / "{target}" / "designs.bed"),
            trajectory=str(DESIGN_DIR / "{target}" / "trajectory.tsv"),
            edits=str(DESIGN_DIR / "{target}" / "edits.tsv"),
        wildcard_constraints:
            target="|".join(re.escape(name) for name in DESIGN_TARGET_NAMES),
        run:
            _concat_files(input.fa, output.fa)
            _concat_files(input.bed, output.bed)
            _merge_tsvs(input.tsv, output.tsv)
            _merge_tsvs(input.trajectory, output.trajectory)
            _merge_tsvs(input.edits, output.edits)
