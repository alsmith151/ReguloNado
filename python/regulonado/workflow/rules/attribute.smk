"""ISM attribution: locate each candidate's high-attribution core (the nucleosome-free region).

Optional stage — absent 'attribution:' config means these rules are never defined and 'rule all'
is unaffected. Cost is ~3 x span x folds forward passes per candidate and candidates are fully
independent, so the candidate BED is sharded across jobs/GPUs and merged back together, exactly
as the design stage does.

This stage is normally upstream of `design`: point `design.candidates` at
`results/attribution/{target}/core_regions.bed` and Snakemake chains the two automatically.
"""

import re
import shlex
from pathlib import Path

ATTRIBUTION = config.get("attribution")
ATTRIBUTION_DIR = RESULTS / "attribution"


def _attr_concat_files(paths, out_path):
    with open(out_path, "w") as out:
        for path in paths:
            out.write(Path(path).read_text())


def _attr_merge_tsvs(paths, out_path):
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


if ATTRIBUTION:

    ATTRIBUTION_RUN_NAMES = ATTRIBUTION.get("runs") or list(RUN_NAMES)
    ATTRIBUTION_TARGETS = ATTRIBUTION["targets"]
    ATTRIBUTION_TARGET_NAMES = [target["name"] for target in ATTRIBUTION_TARGETS]
    ATTRIBUTION_TARGET_BY_NAME = {target["name"]: target for target in ATTRIBUTION_TARGETS}
    ATTRIBUTION_CHECKPOINT_DIRS = ATTRIBUTION.get("checkpoint_dirs")

    _attr_candidates_path = Path(ATTRIBUTION["candidates"])
    # The candidates file may itself be another rule's output, in which case it does not exist
    # at DAG-construction time; fall back to the requested shard count rather than collapsing
    # to a single shard. Empty shards degrade gracefully (resolve_seeds returns []).
    _attr_n_candidates = (
        sum(1 for line in _attr_candidates_path.read_text().splitlines() if line.strip())
        if _attr_candidates_path.exists()
        else 0
    )
    _attr_requested_shards = ATTRIBUTION.get("shards", 1)
    ATTRIBUTION_SHARD_COUNT = (
        max(1, min(_attr_requested_shards, _attr_n_candidates))
        if _attr_n_candidates
        else _attr_requested_shards
    )
    ATTRIBUTION_SHARDS = [str(i) for i in range(ATTRIBUTION_SHARD_COUNT)]

    def _attr_run_dirs():
        return [str(phase_run_dir(run, PHASE_NAMES[-1])) for run in ATTRIBUTION_RUN_NAMES]

    def _attr_checkpoint_state_inputs(wildcards):
        if ATTRIBUTION_CHECKPOINT_DIRS:
            return list(ATTRIBUTION_CHECKPOINT_DIRS)
        return [
            str(phase_run_dir(run, PHASE_NAMES[-1]) / "trainer_state.json")
            for run in ATTRIBUTION_RUN_NAMES
        ]

    def _attr_checkpoint_args():
        if ATTRIBUTION_CHECKPOINT_DIRS:
            return " ".join(
                f"--checkpoint {shlex.quote(str(path))}" for path in ATTRIBUTION_CHECKPOINT_DIRS
            )
        return ""

    # ISM-sweep tuning fields of AttributionConfig; everything else in ATTRIBUTION (candidates,
    # shards, runs, checkpoint_dirs, targets, ...) is structural and handled via rule wildcards.
    _ATTR_SETTINGS_KEYS = {
        "bin_reduction", "topk_bins", "fold_reduction", "pad", "stride", "positions",
        "on_missing", "smooth_bp", "quantile", "min_width_bp", "merge_gap_bp", "min_zscore",
        "max_cores_per_candidate", "anchor", "fix_width", "bigwig", "rtol", "fold_mode",
        "batch_size", "device",
    }

    def _attr_settings_json(wildcards):
        """Merge shared then per-target ISM-sweep settings into one JSON blob for --params.

        ``regulonado attribute`` only takes the handful of I/O flags below directly; everything
        else (bin/fold reduction, smoothing, thresholds, ...) reaches it as a validated
        AttributionConfig loaded from this file.
        """
        import json

        target = ATTRIBUTION_TARGET_BY_NAME[wildcards.target]
        merged = {k: v for k, v in ATTRIBUTION.items() if k in _ATTR_SETTINGS_KEYS}
        merged.update(_flatten_settings(target.get("settings", {})))
        return json.dumps(merged, sort_keys=True)

    rule shard_attribution_candidates:
        input:
            ATTRIBUTION["candidates"],
        output:
            expand(str(ATTRIBUTION_DIR / "shards" / "{shard}.bed"), shard=ATTRIBUTION_SHARDS),
        run:
            # Round-robin, not contiguous blocks, so chromosome clustering in the BED
            # doesn't make one shard dominate the runtime.
            lines = [line for line in Path(input[0]).read_text().splitlines() if line.strip()]
            buckets = [[] for _ in output]
            for index, line in enumerate(lines):
                buckets[index % len(buckets)].append(line)
            for bucket, out_path in zip(buckets, output):
                text = "\n".join(bucket)
                Path(out_path).write_text(text + "\n" if text else "")

    rule attribute_shard:
        input:
            shard=str(ATTRIBUTION_DIR / "shards" / "{shard}.bed"),
            intervals=config["inputs"]["intervals"],
            checkpoints=_attr_checkpoint_state_inputs,
        params:
            track=lambda w: ATTRIBUTION_TARGET_BY_NAME[w.target]["track"],
            fasta=config["inputs"]["fasta"],
            dataset_dir=str(training_dataset_dir()),
            out_dir=lambda w: str(ATTRIBUTION_DIR / w.target / "shards" / w.shard),
            settings_json=_attr_settings_json,
            resolver=str(Path(workflow.basedir) / "scripts" / "resolve_checkpoint.py"),
            run_dirs=_attr_run_dirs(),
            explicit_checkpoint_args=_attr_checkpoint_args(),
        output:
            bed=str(ATTRIBUTION_DIR / "{target}" / "shards" / "{shard}" / "core_regions.bed"),
            cores=str(ATTRIBUTION_DIR / "{target}" / "shards" / "{shard}" / "cores.tsv"),
            summary=str(ATTRIBUTION_DIR / "{target}" / "shards" / "{shard}" / "summary.tsv"),
            attributions=str(
                ATTRIBUTION_DIR / "{target}" / "shards" / "{shard}" / "attributions.tsv"
            ),
            run_json=str(ATTRIBUTION_DIR / "{target}" / "shards" / "{shard}" / "run.json"),
        resources:
            gpu=1,
            runtime=240,
        wildcard_constraints:
            target="|".join(re.escape(name) for name in ATTRIBUTION_TARGET_NAMES),
            shard="|".join(re.escape(s) for s in ATTRIBUTION_SHARDS),
        log:
            str(RESULTS / "logs" / "attribute_{target}_{shard}.log"),
        shell:
            r"""
            set -euo pipefail

            CHECKPOINT_ARGS=()
            if [ -n "{params.explicit_checkpoint_args}" ]; then
                CHECKPOINT_ARGS=({params.explicit_checkpoint_args})
            else
                for run_dir in {params.run_dirs}; do
                    CKPT=$(python {params.resolver:q} "$run_dir")
                    CHECKPOINT_ARGS+=(--checkpoint "$CKPT")
                done
            fi

            PARAMS_FILE=$(mktemp)
            trap 'rm -f "$PARAMS_FILE"' EXIT
            printf '%s' {params.settings_json:q} > "$PARAMS_FILE"

            regulonado attribute \
                --params "$PARAMS_FILE" \
                --candidates {input.shard:q} \
                --intervals {input.intervals:q} \
                "${{CHECKPOINT_ARGS[@]}}" \
                --fasta {params.fasta:q} \
                --dataset-dir {params.dataset_dir:q} \
                --track {params.track:q} \
                --out {params.out_dir:q} \
                > {log:q} 2>&1
            """

    rule merge_attributions:
        input:
            bed=lambda w: expand(
                str(ATTRIBUTION_DIR / w.target / "shards" / "{shard}" / "core_regions.bed"),
                shard=ATTRIBUTION_SHARDS,
            ),
            cores=lambda w: expand(
                str(ATTRIBUTION_DIR / w.target / "shards" / "{shard}" / "cores.tsv"),
                shard=ATTRIBUTION_SHARDS,
            ),
            summary=lambda w: expand(
                str(ATTRIBUTION_DIR / w.target / "shards" / "{shard}" / "summary.tsv"),
                shard=ATTRIBUTION_SHARDS,
            ),
            attributions=lambda w: expand(
                str(ATTRIBUTION_DIR / w.target / "shards" / "{shard}" / "attributions.tsv"),
                shard=ATTRIBUTION_SHARDS,
            ),
        params:
            fasta=config["inputs"]["fasta"],
        output:
            bed=str(ATTRIBUTION_DIR / "{target}" / "core_regions.bed"),
            cores=str(ATTRIBUTION_DIR / "{target}" / "cores.tsv"),
            summary=str(ATTRIBUTION_DIR / "{target}" / "summary.tsv"),
            attributions=str(ATTRIBUTION_DIR / "{target}" / "attributions.tsv"),
            bigwig=str(ATTRIBUTION_DIR / "{target}" / "attributions.bw"),
        wildcard_constraints:
            target="|".join(re.escape(name) for name in ATTRIBUTION_TARGET_NAMES),
        run:
            _attr_concat_files(input.bed, output.bed)
            _attr_merge_tsvs(input.cores, output.cores)
            _attr_merge_tsvs(input.summary, output.summary)
            _attr_merge_tsvs(input.attributions, output.attributions)

            # Rebuild one BigWig from the merged per-base table rather than trying to splice the
            # per-shard BigWigs; shards hold disjoint candidates but can still overlap genomically.
            from regulonado.design.attribution import merge_attribution_bigwig

            merge_attribution_bigwig(
                output.attributions, output.bigwig, fasta_path=params.fasta
            )
