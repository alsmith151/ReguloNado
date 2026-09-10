# ReguloNado repository audit

**Audit date:** 2026-09-10  
**Scope:** the current working tree, including uncommitted and untracked files  
**Purpose:** simplify and modernise the existing system without gratuitous rewrites

## Executive summary

ReguloNado has a sound high-level shape and much more behavioural coverage than its alpha version number suggests. The central design is appropriate for the problem: Rust performs the expensive BigWig/FASTA-to-Arrow work; Python owns discovery, validation, orchestration, model training, inference, and design; Typer exposes interactive commands; Hydra composes training presets; and Snakemake schedules the multi-stage workflow. Those boundaries should remain.

The main problem is not excessive architecture. It is that several large orchestration functions have accumulated too many responsibilities, while a few file-operation and error-handling details violate the system's scientific and durability guarantees. Four issues should stop a release:

1. A partial incremental dataset build can delete splits that were deliberately skipped.
2. In-place recompression with `overwrite=True` can delete the only source dataset.
3. Rust BigWig reads convert all interval-read failures into valid-looking zero signal.
4. The declared `seqnado>=1.1` extra makes the project unsatisfiable to uv because no such PyPI release exists; the checked-in lock file no longer represents the manifest.

The next tier is operational rather than cosmetic: fixed shared scratch directories race between builds; recompression loads a complete Arrow shard into memory despite claiming to stream; whole-genome prediction accumulates every output interval in memory; QC silently ignores misspelled metrics; and training validation is not applied at the entry point that actually trains.

The best route is a controlled contraction, not a rewrite. First add regression tests around the destructive and silent-failure paths, then make file publication transactional, give every invocation private scratch space, fix dependency resolution, delete approximately 175 lines of unused legacy dataset-reader code, remove dead parameters, and remove dependencies for which there is no production use. Only after those invariants are protected should the CLI and training runner be split into thin command adapters and independently testable services.

### Candid overall judgement

- The algorithms and domain decomposition are broadly credible.
- The happy paths work: the full local Python suite passes, Rust formats and lints cleanly, and the CLI loads.
- The failure paths are less mature than the happy paths. Two are destructive and one can silently corrupt scientific output.
- The dependency manifest and developer instructions are currently inconsistent enough that a clean uv setup cannot be reproduced.
- The project needs fewer public concepts and fewer unused packages, not another framework.

## Evidence and validation

This review inspected the repository structure, Python and Rust sources, configuration models and generated schema, Typer commands, Hydra presets, Snakemake workflow, documentation, examples, tests, packaging, and GitHub Actions. Comments and documentation were checked against implementation rather than treated as authoritative.

Local checks on the audited working tree:

| Check | Result | Interpretation |
| --- | --- | --- |
| `.venv/bin/pytest tests -q` | **258 passed, 9 skipped** in about 39 s | Strong happy-path coverage. Every skip was optional SeqNado integration because SeqNado was unavailable. |
| `.venv/bin/ruff check python/ tests/` | **Failed:** one `I001` in `design/diagnostics.py:2` | Small issue, but CI would fail on the current tree. |
| `cargo fmt --check` | **Passed** | Rust formatting is clean. |
| `cargo clippy --all-targets -- -D warnings` | **Passed** | Rust lint baseline is good. |
| `cargo test --all-targets` | **Passed, 0 tests executed** | The native core has no Rust unit tests. Python parity tests do not replace native failure-path tests. |
| `.venv/bin/python -m regulonado --help` | **Passed** | Installed CLI entry point and lazy core import work. |
| `git diff --check` | **Passed** | No whitespace damage in the current changes. |
| `uv lock --check` / `uv run pytest` | **Resolution failed** | `seqnado>=1.1` is unavailable from PyPI; `uv.lock` is stale relative to `pyproject.toml`. |

The checkout was already substantially dirty. This audit does not attribute those changes to a particular author and does not assume they have been committed or released. Findings refer to the working tree that was tested.

## Repository overview

The tracked repository is modest in size: roughly 13,000 lines of product Python, 1,500 lines of Rust, and 5,400 lines of Python tests across 51 Python modules, 8 Rust source files, and 19 test modules. The complexity is concentrated rather than evenly distributed:

| Area | Approximate size / concentration | Role |
| --- | --- | --- |
| `python/regulonado/cli/app.py` | 2,422 lines | Most user commands and substantial orchestration/business logic |
| `training/runner.py` | 1,285 lines | Dataset/model/trainer construction and execution |
| `model/heads.py` | 1,133 lines | Domain-specific neural heads |
| `dataset/build.py` | 1,061 lines | Dataset planning, staging, native dispatch, publication, transforms |
| `tracks.py` | 772 lines; `TrackSheet` about 575 lines | Track discovery and metadata ingestion |
| `inference.py` | 764 lines | Model restoration, regional and genome-wide prediction |
| `design/attribution.py` | 718 lines | Attribution implementation and output |
| `src/chromosome_scan_writer.rs` / `src/binning.rs` | Native hot path | Chromosome-pass BigWig binning, FASTA extraction, Arrow writing |

### Actual architecture and data flow

```text
User
  |
  +-- `regulonado <command>` (Typer)
  |      +-- track discovery / QC / normalisation
  |      +-- direct dataset build, prediction, attribution, design
  |      +-- single training invocation through Hydra
  |      `-- Snakemake pipeline launcher
  |
  +-- Python modules (usable but not presented as a stable facade)

Track sources / SeqNado projects
  -> discovery and fingerprints
  -> optional scale factors + QC metrics
  -> canonical `tracks.parquet`
  -> dataset builder
       -> Rust PyO3 extension
            BigWig + FASTA + BED -> compressed Arrow shards
       -> Hugging Face Dataset metadata
  -> optional recompression
  -> transforms / training
       Hydra composition -> Hugging Face Trainer -> checkpoint + provenance
  -> checkpoint restoration
       regional prediction / BigWig prediction / attribution / sequence design

Snakemake owns the multi-stage DAG and restart semantics.
Hydra owns compositional training presets.
Typer owns user-facing command parsing.
Pydantic owns workflow configuration validation and schema generation.
```

### Intended versus actual architecture

The intended architecture in the README is close to reality: one canonical track table feeds a native dataset build and later training. The deviations are important:

- CLI commands frequently contain service logic rather than adapting arguments to application functions. The 554-line `design` command and 320-line `attribute` command are the clearest examples.
- Pydantic is intended to be the workflow authority, but the Snakefile duplicates and contradicts one cross-field rule.
- The dataset builder presents itself as incremental, but publication treats the scratch tree as a complete replacement.
- Recompression presents itself as bounded-memory and safe to overwrite, but neither claim holds for important inputs.
- The top-level Python package exports neural implementation details but omits the common dataset and inference operations, so the effective public API is accidental.

### Core versus incidental complexity

Genuinely core concepts are the canonical track table, coordinate geometry, chromosome-pass writer, signal transforms, model/checkpoint contract, prediction windows, attribution/design objectives, and workflow DAG. The complexity of genomic coordinate handling and model adaptation is real and should not be hidden behind generic abstractions.

Incidental complexity includes fixed scratch paths, manual tree publication, dead pre-Rust readers, no-op compatibility parameters, dual logging systems, a wildcard dataset facade, repeated CLI orchestration, and dependencies imported nowhere in production.

## Prioritised findings

### F1 — Partial incremental builds can delete completed dataset splits

**Priority:** P0 — Critical  
**Evidence:** `dataset/build.py:606-637`, `:654-660`, `:715-721`, and `:763-769`

**Problem.** Existing split directories are skipped when their `dataset_info.json` exists. Only rebuilt splits are placed in the fixed scratch output. The final `_rsync_tree(scratch_out, output_dir, delete=True)` then removes destination entries absent from scratch. In a partial build, skipped splits are absent by design, so `--delete` can erase them.

**Why it matters.** This defeats restartability and can destroy expensive completed work during the operation advertised as incremental. Unit coverage of `_rsync_tree(delete=True)` does not test the cross-function invariant.

**Recommendation.** Publish each rebuilt split independently and atomically. Write global metadata only after all requested splits are present and verified. Never mirror-delete a partial staging tree onto a complete dataset.

**Preferred implementation.** Give each invocation a private staging directory next to the destination where possible. For every rebuilt split: write to `.<split>.<random>.tmp`, validate that Arrow shards and Hugging Face metadata load, then use `os.replace` to publish it. When replacing an existing non-empty directory, use a short backup/rename sequence because POSIX cannot atomically replace a populated directory. Regenerate `dataset_dict.json` from the final destination.

**Alternatives considered.** Copying skipped splits into scratch is correct but duplicates data and I/O. Removing `--delete` avoids immediate loss but leaves obsolete shards and stale splits, so it does not provide a coherent publication contract.

**Web research.** Python documents `os.replace` as an atomic successful rename on the same filesystem and provides `tempfile.TemporaryDirectory` for private, automatically cleaned staging: [os.replace](https://docs.python.org/3/library/os.html#os.replace), [tempfile](https://docs.python.org/3/library/tempfile.html).

**Risk and migration.** Medium implementation risk because directory replacement and cross-filesystem outputs need explicit tests. Keep the on-disk dataset format unchanged. First add a regression fixture with one existing and one missing split; then replace only the publication step.

### F2 — Recompression can delete its own source and publishes partial output

**Priority:** P0 — Critical  
**Evidence:** `recompress.py:84-104`

**Problem.** If destination exists and overwrite is enabled, the destination is recursively removed before any relationship between source and destination is checked. When `src == dst`, the sole dataset is deleted and an empty directory is recreated. Nested source/destination relationships are also unsafe, especially with source removal. Output is written directly to its final location, so a failed shard leaves a partial dataset.

**Why it matters.** The API accepts a normal-looking in-place invocation that irreversibly loses data. Interruptions can also leave output that looks present but is incomplete.

**Recommendation.** Resolve both paths and reject equality or ancestor/descendant relationships. Always write to a unique sibling temporary directory, validate the complete result, publish by rename, and only then honour optional source deletion.

**Preferred implementation.** Introduce a small internal `validate_distinct_trees(src, dst)` and a transactional directory-publication helper shared with dataset building. Preserve the current command name and output format; change invalid destructive invocations to a clear `ValueError`/Typer usage error.

**Alternatives considered.** Supporting true in-place recompression through a temporary sibling is possible, but it complicates recovery and should be a separate explicitly named mode. A confirmation prompt is insufficient for non-interactive workflows and does not make the algorithm safe.

**Web research.** `shutil` documents recursive copy/removal primitives but does not make a multi-file transformation transactional; same-file and metadata semantics must be handled explicitly: [Python shutil documentation](https://docs.python.org/3/library/shutil.html).

**Behavioural change.** Yes: formerly accepted same/nested paths will fail before mutation. That is an intentional safety fix.

**Risk and migration.** Low-to-medium. Add equality, symlink-alias, nested path, mid-stream exception, overwrite, and successful source-removal tests before changing the implementation.

### F3 — BigWig read errors silently become biologically valid zero signal

**Priority:** P0 — Critical  
**Evidence:** `src/binning.rs:79-96` and `:143-146`

**Problem.** `reader.get_interval(...)` errors are converted to an empty result and errors while iterating intervals are skipped. Missing coverage legitimately maps to zeros, but corrupt input, malformed blocks, I/O errors, and incompatible chromosome access can reach the same outcome.

**Why it matters.** A failed measurement becomes a valid-looking label. Training can succeed on silently corrupted tracks, making this a scientific-correctness issue rather than merely poor diagnostics. Python parity tests exercise two writers that share this helper, so both can agree on the same wrong behaviour.

**Recommendation.** Propagate BigWig reader errors with track path, chromosome, start, and end context. Preserve zeros only for the backend's explicit no-interval/no-coverage result. Abort the split rather than publishing incomplete output.

**Preferred implementation.** Change the binning helper to return `Result`; use `?` with contextual errors through the writer boundary. Map invalid user coordinates to `PyValueError`, I/O/backend failures to `PyOSError` or a small custom `RegulonadoDataError`, and internal invariant failures to `PyRuntimeError`.

**Alternatives considered.** A `strict=False` mode would preserve the dangerous default and complicate reproducibility. If best-effort processing is ever required, make it opt-in and emit a machine-readable per-track failure manifest; do not substitute zeros silently.

**Web research.** PyO3 supports Python built-in exception mappings and custom exception classes: [PyO3 exception handling](https://pyo3.rs/main/exception), [PyValueError](https://pyo3.rs/main/doc/pyo3/exceptions/struct.pyvalueerror). The underlying bigtools project is an appropriate high-performance Rust/Python BigWig foundation and need not be replaced: [bigtools](https://github.com/jackh726/bigtools).

**Risk and migration.** Medium: backend error variants need classification. Add Rust unit tests and Python integration tests with a truncated BigWig, a missing chromosome, and injected read failure before changing propagation.

### F4 — The declared environment cannot currently be resolved reproducibly

**Priority:** P0 for release readiness; P1 for runtime behaviour  
**Evidence:** `pyproject.toml` `seqnado` extra, `[tool.uv]`, `CONTRIBUTING.md`; stale `uv.lock`

**Problem.** The manifest declares `seqnado>=1.1`, while PyPI's latest published release is 1.0.7 and the code comment says the required API exists only on a development branch. uv resolves optional dependencies as part of the universal project solution, so even developers who did not request that extra cannot generate a current lock. The checked-in lock lacks several current core dependencies and SeqNado entirely, while the contribution guide tells users to run `uv sync`.

**Why it matters.** A clean contributor cannot follow the documented setup, the lock is not a reproducible description of the project, and CI's pip installs do not detect uv lock drift.

**Recommendation.** Do not publish a dependency constraint that no index release can satisfy. Until SeqNado 1.1 exists, either remove the published `seqnado` extra and document an explicit development-branch install, or use a uv-only Git source for development while ensuring built metadata remains honest. Constrain uv's supported environments to the project's actual Linux/macOS classifiers if Windows-only resolution contributes conflicts. Regenerate and commit the lock.

**Preferred implementation.** Remove the unsatisfiable optional extra from distribution metadata for the next release, add a documented `uv add --optional seqnado "seqnado @ git+...@<commit>"` development route only if a pinned commit is acceptable, and restore the extra when a compatible PyPI release exists. Add `uv lock --check` to CI.

**Alternatives considered.** Declaring `seqnado @ git+...@develop` in published metadata makes builds depend on mutable VCS state and is inferior to a commit pin. Lying with `>=1.0.7` would install an API the code explicitly says is absent.

**Web research.** uv documents its universal resolution across extras and the `tool.uv.environments` restriction: [uv resolution](https://docs.astral.sh/uv/concepts/resolution/), [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/). Registry evidence: [SeqNado on PyPI](https://pypi.org/project/seqnado/). Direct URL dependency syntax is standard packaging metadata: [PyPA dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/).

**Behavioural change.** Packaging only. Existing manually installed development SeqNado integrations continue to use optional runtime imports.

### F5 — Shared fixed scratch directories race and are not failure-isolated

**Priority:** P1 — High  
**Evidence:** `dataset/build.py:496-505`, `:646`, `:671-674`, `:731-734`, `:769`

**Problem.** Every invocation uses `$TMPDIR/regulonado_build` and, optionally, `$TMPDIR/regulonado_stage`. A second process, concurrent Snakemake job, or another user sharing a configured scratch root can delete or overwrite the first process's files.

**Why it matters.** The workflow is designed for concurrent/cluster execution. Fixed names create nondeterministic corruption, misleading failures, and unsafe cleanup.

**Recommendation.** Allocate a unique per-invocation temporary directory. Treat source staging as invocation-local unless a separately designed, content-addressed, locked cache is needed.

**Preferred implementation.** Use `TemporaryDirectory(prefix="regulonado-build-", dir=scratch_root)` and a sibling stage subdirectory within it. Allow an explicit scratch parent, not an explicit shared working directory.

**Alternatives considered.** PID suffixes still collide after PID reuse and leak on failure. File locks serialize builds unnecessarily and do not solve stale partial contents. A shared cache is a separate feature requiring immutable keys and locking.

**Web research.** `TemporaryDirectory` uses securely generated names, supports a chosen parent, and provides deterministic context cleanup: [Python tempfile documentation](https://docs.python.org/3/library/tempfile.html).

**Risk and migration.** Low. Tests should launch two small builds concurrently and assert disjoint staging paths and complete outputs.

### F6 — Recompression is not actually streaming

**Priority:** P1 — High  
**Evidence:** `recompress.py:25-29` and its bounded-memory documentation

**Problem.** `batches = list(reader)` materialises every record batch in a shard before opening the writer. Memory therefore scales with the full uncompressed shard.

**Why it matters.** Genomic label matrices are large; a maintenance operation intended to reduce storage can exhaust memory. The implementation contradicts its documentation.

**Recommendation.** Keep reader and writer open together and forward one `RecordBatch` at a time. Combine this with F2's transactional output.

**Preferred implementation.** Read the schema once, construct the output stream, iterate `for batch in reader: writer.write_batch(batch)`, then close both through context managers.

**Alternatives considered.** Reducing shard size treats the symptom and increases file-count overhead. Hugging Face round-tripping adds an unnecessary abstraction and may rewrite schema metadata.

**Web research.** Arrow's IPC API is explicitly batch-oriented and documents reading/writing record batches as a stream: [Apache Arrow IPC](https://arrow.apache.org/docs/python/ipc.html).

**Risk and migration.** Low. Compare schema, row count, batch-independent content hashes, loadability, and compression settings before/after.

### F7 — Configuration has two authorities and they disagree

**Priority:** P1 — High  
**Evidence:** `config/models.py:353-375`; `workflow/Snakefile:20-38`; `config/generator.py:225-237`

**Problem.** The Pydantic comment says an explicit SeqNado scaling project cannot make multiple aggregated projects valid, but its condition rejects multiple projects only when the explicit field is absent. The Snakefile rejects all multiple-project cases. A config can therefore pass the advertised model and fail only after Snakemake starts. Separately, the generator offers `tmm`, `original`, `bamnado`, and optionally `seqnado`, but omits the supported `anchor` method.

**Why it matters.** Users see different validity rules depending on entry point, and the safest way to create a configuration cannot express a supported feature.

**Recommendation.** Make `RegulonadoConfig` the sole semantic authority. Validate the final YAML before invoking Snakemake and make workflow rules consume already validated fields. Fix the multi-project condition and add the anchor prompts/requirements to the generator.

**Preferred implementation.** Have the pipeline command call `RegulonadoConfig.from_yaml`, then pass its normalised dump to Snakemake. Retain Snakemake's schema validation as an early structural check but remove duplicated cross-field Python from the Snakefile. Generate JSON Schema from Pydantic as the repository already does.

**Alternatives considered.** Snakemake supports JSON Schema defaults and validation, but JSON Schema is awkward for these cross-object semantic rules; Pydantic already owns them. Moving everything into the Snakefile would make direct Python/CLI validation worse.

**Web research.** Pydantic generates JSON Schema from the same typed model: [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/). Snakemake documents schema-based configuration validation and default population: [Snakemake configuration validation](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation).

**Behavioural change.** Yes: the explicit-project/multiple-project combination will fail earlier, matching the existing workflow and comment.

### F8 — Public build parameters are accepted but do nothing

**Priority:** P1 — High  
**Evidence:** `dataset/build.py:407-435`, assignment at `:496-498`; CLI options in `cli/app.py:1208-1265` and call at `:1367-1393`

**Problem.** `signal_sample_chunk`, `signal_track_chunk`, `num_proc`, and `writer_batch_size` are accepted and forwarded but not used by the current writer. `cache_dir` is assigned a default and then unused. The CLI defines `io_threads` but does not pass it to the builder at all.

**Why it matters.** Operators can tune values believing they affect memory or throughput. No-op controls are worse than absent controls because they invalidate operational reasoning and complicate tests/documentation.

**Recommendation.** Remove the options from the primary API. For one release, accept them only through a deprecated compatibility wrapper that warns they have no effect. Rename `build_dataset_fast` to `build_dataset`; keep the old name as a warning alias.

**Preferred implementation.** Define a small `DatasetBuildOptions` dataclass containing only active fields if the service is called from multiple adapters; otherwise keep explicit keyword arguments. Do not introduce a general builder object.

**Alternatives considered.** Reimplementing behaviour merely to justify the parameters is backwards reasoning. Silently retaining them preserves a false contract.

**Behavioural change.** The options already have no behaviour. The observable change is a deprecation warning followed by a future signature error.

### F9 — QC configuration typos silently disable rules

**Priority:** P1 — High  
**Evidence:** `qc.py:217-239`; explicitly codified by `tests/test_track_qc.py:85-89`; `config/models.py:160`

**Problem.** `evaluate_rules` skips a configured column not present in the metrics table and reports every row as passed. The schema permits arbitrary rule keys, so `qc_nonzero_bin_fracton` silently disables the intended gate.

**Why it matters.** QC is a scientific acceptance boundary. A typo must not turn a failed/unknown check into a pass.

**Recommendation.** Validate rule keys against the columns produced by the enabled checks and fail with the unknown key plus valid alternatives. Also reject `min > max` and rules for disabled check families.

**Preferred implementation.** Maintain one constant mapping from check family to metric names in `qc.py`; use it in both Pydantic validation and the evaluator. The evaluator should defensively raise if a key is still missing.

**Alternatives considered.** Warnings are easy to miss in batch logs and still allow an invalid dataset. Strict DataFrame schemas alone cannot know which optional checks were enabled.

**Behavioural change.** Yes: formerly ignored rules become configuration errors. This is desirable and should be noted in release notes.

### F10 — Training bypasses full structured validation on its real execution path

**Priority:** P1 — High  
**Evidence:** `training/runner.py:960-1255` and `:1258-1273`; trainer-only validation at `:1108-1115`

**Problem.** `run_training` accepts a nested `Mapping[str, Any]` and performs a 296-line orchestration using string keys. The Hydra entry point resolves to a plain dict and calls it directly. A separate resolved/print configuration path validates more structure, but actual training only converts the trainer subsection to a dataclass late in execution.

**Why it matters.** Misspelled or inconsistent fields fail after expensive dataset/model setup, with `KeyError` or component-specific errors. Static analysis cannot describe the actual contract.

**Recommendation.** Validate one `TrainingConfig` at the Hydra boundary and pass it to smaller services: dataset loading, model construction, trainer construction, execution, and result persistence. Keep Hydra for composition; it is not the problem.

**Preferred implementation.** Use OmegaConf structured dataclasses for the whole resolved training configuration, because this module already uses them. Make `run_training(config: TrainingConfig, ...)` the stable service. A transitional adapter can accept mappings, validate immediately, and warn.

**Alternatives considered.** Duplicating equivalent Pydantic models would add a second training schema system. Replacing Hydra would discard useful preset composition without fixing orchestration.

**Risk and migration.** Medium. Snapshot resolved configs for all packaged presets, add unknown/missing-key tests, then change the internal signature behind the CLI.

### F11 — Whole-genome inference has unbounded output accumulation and unsafe names

**Priority:** P2 for normal regional prediction; P1 for advertised whole-genome runs  
**Evidence:** `inference.py:611-728` and `:731-750`

**Problem.** `predict_to_bigwig` appends every collapsed interval for every selected track to `accum`, then sorts everything before writing. Memory grows with genome size times track count. Output paths interpolate checkpoint-derived track names directly into filenames; the lower-level track-table schema does not guarantee safe filename components for all direct callers.

**Why it matters.** Whole-genome inference can exhaust RAM after the expensive model work finishes. Crafted or legacy metadata containing separators can write outside the intended directory. The latter is a confirmed missing validation and a potential path traversal, not evidence of a known exploit.

**Recommendation.** Process in chromosome order and flush each chromosome to a bounded intermediate representation, or keep one writer/spool per selected track and merge adjacent intervals incrementally. Convert display names to collision-resistant safe filenames and verify `path.resolve().parent == out_dir.resolve()`.

**Preferred implementation.** First spool per-track, per-chromosome Arrow/temporary interval files in a private temporary directory; after prediction, stream them in header order to BigWig writers. This bounds RAM without requiring undocumented incremental behaviour from pybigtools. Store a JSON mapping from original track name to filename.

**Alternatives considered.** One open BigWig writer per track may exhaust file descriptors and depends on writer streaming semantics that should be verified against the installed pybigtools version. Writing each track in a separate model pass bounds output memory but multiplies inference cost.

**Risk and migration.** Medium. Golden tests must compare intervals byte-semantically, including chromosome order, overlap collapse, filename collisions, Unicode, and separator input.

### F12 — The dependency surface is substantially larger than actual production use

**Priority:** P1 — High  
**Evidence:** `pyproject.toml` plus repository-wide import search

**Problem.** `polars` has no product or test imports. `pysam`, `psutil`, `lightning`, `peft`, `plotly`, and `seaborn` have no production imports. `tangermeme` is deliberately not used by the attribution implementation. `bioframe` is used only to read a BED-like table. Loguru and standard-library logging coexist, with repeated `basicConfig` calls in CLI functions.

**Why it matters.** Data and training environments are already large and platform-sensitive. Every direct dependency increases resolution, installation, vulnerability, licensing, and compatibility work. It also falsely communicates architecture to contributors.

**Recommendation.** Remove unused direct dependencies after an import/build matrix proves they are not required transitively at build time. Replace the one `bioframe.read_table(..., schema="bed")` use with the already-required pandas parser. Standardise on stdlib logging and configure it once at the CLI boundary. Keep specialised scientific packages that are actually exercised.

**Preferred implementation.** See the dependency table below. Make removals one logical group at a time and test clean wheels rather than trusting the current environment.

**Alternatives considered.** Moving notebook-only packages into `viz`/`jupyter` is acceptable if maintained examples really import them; otherwise deletion is clearer. Keeping Loguru would be reasonable if structured sinks, contextual binding, or exception decoration were used consistently, but they are not.

**Web research.** pandas already supports delimiter, selected columns, comments, types, and malformed-row policy in `read_csv`: [pandas.read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html). Stdlib logging uses hierarchical module loggers and central handler configuration: [Python logging](https://docs.python.org/3/library/logging.html). Registry/project checks show that [Polars](https://pypi.org/project/polars/), [Tangermeme](https://pypi.org/project/tangermeme/), [Bioframe](https://pypi.org/project/bioframe/), and [Loguru](https://github.com/Delgan/loguru) are active projects; removal is based on fit and use, not abandonment.

**Risk and migration.** Low-to-medium. The risk is undeclared runtime use in external notebooks. Search and run packaged examples, document the reduced extras, build wheels, and install each supported extra into a clean environment.

## Secondary findings by area

### Architecture and complexity

| Priority | Finding | Evidence | Concrete action |
| --- | --- | --- | --- |
| P2 | The Typer app is a god module; command functions mix argument presentation, loading, computation, progress, and persistence. | `cli/app.py` (2,422 lines); `design` ~554 lines; `attribute` ~320; build ~229 | Follow the existing `cli/tracks.py` precedent. Create `cli/build.py`, `cli/predict.py`, `cli/design.py`, `cli/train.py`; move reusable work into domain services and keep command bodies near 20–60 lines. |
| P2 | `TrackSheet` combines parsing from several origins, deduplication, metadata transformations, and persistence. | `tracks.py`, class around 575 lines | Preserve `TrackSheet` as the user abstraction, but extract source adapters (`from_directory`, SeqNado, CSV) into pure functions returning canonical records. Do not introduce interfaces/factories. |
| P2 | `dataset.__init__` dynamically copies every name, including private helpers, from two modules. | `dataset/__init__.py` | Replace wildcard global mutation with an explicit `__all__`. Preserve genuinely used historical imports with deprecation aliases. |
| P2 | Four pre-native dataset helpers are uncalled legacy code. | `dataset/build.py`: `_read_track`, `_read_all_tracks`, `_is_contiguous`, `_sample_major_signal_generator` | Delete after confirming no external support commitment; approximately 175 lines plus imports and dead chunk controls can go. |
| P3 | `scaling.py` is a three-line wildcard compatibility facade. | `scaling.py` | Search downstream use/releases. If never public, delete; otherwise give one deprecation cycle and explicit exports. |
| P2 | Module naming exposes implementation history (`build_dataset_fast`) rather than present behaviour (`chrom_pass` is default). | `dataset/build.py:407` | Introduce `build_dataset`; retain a deprecated alias. |

### API and usability

The CLI is the documented primary interface and is reasonably discoverable through Typer. The Python API is not. `regulonado.__init__` lazily exports many neural head/adapter implementation classes but does not expose `build_dataset`, `RegionPredictor`, or `load_model_for_inference`. This is inverted for a new user.

Choose and document one of two contracts:

1. **Recommended for 0.x:** the CLI and file formats are stable; the Python API is experimental except for a deliberately small facade.
2. Declare all currently exported model internals public and accept the long-term compatibility cost. This is not recommended.

The small facade should expose common workflows and typed results, while advanced users can continue importing submodules.

Current dataset usage:

```python
from regulonado.dataset import build_dataset_fast

dataset = build_dataset_fast(
    bed_file,
    fasta_file,
    tracks_parquet,
    output_dir,
    signal_sample_chunk=128,  # has no effect
    writer_batch_size=500,    # has no effect
)
```

Proposed usage:

```python
from regulonado import build_dataset

result = build_dataset(
    regions=bed_file,
    reference=fasta_file,
    tracks=tracks_parquet,
    output=output_dir,
)
print(result.splits, result.track_count, result.sample_count)
```

`DatasetBuildResult` should be a frozen dataclass containing paths/counts and optionally a lazily loaded `DatasetDict`; do not make a builder class. The existing return of `object | None` is undiscoverable. For compatibility, `return_dataset=True` can remain in the old alias during migration.

Current inference usage requires internal knowledge and a mandatory dataset directory even though modern checkpoints aim to be self-contained:

```python
from regulonado.inference import RegionPredictor

predictor = RegionPredictor.from_checkpoint(checkpoint, fasta, dataset_dir=dataset)
prediction = predictor.predict("chr1", 1_000_000, 1_001_000, tracks=["H3K27ac"])
```

Proposed common case:

```python
from regulonado import RegionPredictor

with RegionPredictor.open(checkpoint, reference=fasta) as predictor:
    prediction = predictor.predict("chr1:1000000-1001000", tracks=["H3K27ac"])
```

Keep the tuple form too; the string form is convenience, not a replacement. `dataset_dir` should be optional when checkpoint metadata is complete and required with a precise error only for legacy checkpoints. A context manager makes FASTA and any native resources' lifetime explicit.

For training, prefer a validated object over a nested mapping:

```python
config = TrainingConfig.from_hydra(cfg)
summary = run_training(config)
```

Do not expose separate objects for every internal phase. `TrainingSummary` can wrap the current dictionary without changing its saved JSON representation.

### Error handling and durability

| Priority | Finding | Evidence | Recommendation |
| --- | --- | --- | --- |
| P2 | Rust boundary errors are predominantly generic `PyRuntimeError`. | PyO3 functions under `src/` | Map invalid arguments, filesystem/backend errors, and invariant failures to stable categories; attach path/coordinate context. Avoid a deep custom hierarchy. |
| P2 | `write_track_table` uses a predictable fixed `.tmp` path, so concurrent writers collide. | `tracks_table.py:115-139` | Use `NamedTemporaryFile` in the destination directory, close, then `os.replace`; clean it in `finally`. |
| P2 | BED parsing indexes raw fields and calls `int` without line/domain validation. | `dataset/build.py:_load_bed_rows`; `inference.py:_parse_bed` | Share a small parser returning typed intervals with filename/line errors; reject negative starts, `end <= start`, missing fold, and overflow before native work. |
| P2 | BigWig/FASTA resource closure is inconsistent. | QC uses explicit `finally`; inference retains FASTA; writer close is implicit | Define context ownership in APIs. Verify pybigtools context-manager support for the pinned version; otherwise close explicitly in `finally`. |
| P3 | External command errors are generally surfaced, but timeout/cancellation policy is undocumented. | bamnado/rsync/Snakemake subprocess paths | Do not add arbitrary short timeouts to legitimate cluster jobs. Add process context to errors and forward cancellation/signals; support configurable timeout only for bounded helper commands. |

### Testing

What is good:

- The 258 passing tests are fast enough to run locally.
- Chromosome-pass and direct writers have parity coverage.
- Track-table, fingerprint, model-transfer, loss, metric, workflow-DAG, design, attribution, prediction, and configuration behaviours are exercised.
- Tests mostly validate public behaviour rather than chasing line coverage.
- Model tests avoid network/pretrained downloads through dummy adapters.

Important missing coverage, in order:

1. Partial incremental publication preserving an existing split.
2. Recompression equality, nesting, interruption, and transactional overwrite.
3. Corrupt/missing-chromosome BigWig failures through Rust and Python.
4. Two concurrent builds using the same scratch parent.
5. Full resolved training config validation on the actual entry point.
6. QC unknown metric and invalid bounds rejection.
7. Whole-genome inference memory behaviour and safe filename collisions.
8. Clean wheel installation for core/data/train/workflow extras on Python 3.12 and 3.13.
9. Native Rust unit tests for bin geometry, interval clipping, strand reversal, missing coverage, and backend errors.

Property-based testing is justified only for compact invariant-heavy code. Good targets are interval clipping/binning and `transform_signal`/`inverse_transform_signal` round trips. [Hypothesis](https://hypothesis.readthedocs.io/) is mature, but adding it is optional: a small deterministic parameter grid may cover these invariants with less dependency surface. Fuzzing arbitrary CLI/config inputs is lower value than the failure tests above.

Do not pursue 100% coverage. Add tests where a wrong result is destructive, silent, or scientifically plausible.

### Documentation

The repository has useful focused guides for datasets, checkpoints, prediction, attribution, design, normalisation, Slurm, SeqNado, QC, and track tables. The README provides a real pipeline diagram and commands. The main issues are consistency and progressive disclosure:

- README `:127-132` says QC is a workflow stage, while `:137-138` says the pipeline does not perform QC.
- `CONTRIBUTING.md` recommends `uv sync`, which cannot currently resolve the manifest.
- `normalization.md` says regions are read with Bioframe; that becomes stale if the proposed dependency deletion is accepted.
- Public Python stability and exceptions are not documented.
- Operational recovery semantics for build/recompress are absent—and currently unsafe.
- Configuration models and generated schema are the real authority, but the `config/models.py` module docstring calls them a mirror.

Recommended documentation hierarchy:

1. README: install one supported mode, five-minute dataset/training path, link outward.
2. `docs/concepts.md`: track table, regions/bins, dataset, checkpoint, scaling/QC.
3. Task guides: build, train, predict, attribute, design, pipeline/cluster.
4. `docs/configuration.md`: generated field reference plus three complete examples.
5. `docs/python-api.md`: explicit stability statement and common facade only.
6. `docs/troubleshooting.md`: input validation, native errors, recovery, memory sizing.
7. Contributor architecture/testing guide.

Avoid duplicating every option manually in prose. Generate reference tables from Pydantic/Typer where practical, then maintain narrative examples by hand.

### Performance

**Clearly incorrect inefficient behaviour:** full-shard recompression materialisation and whole-genome output accumulation. Fix both.

**Credible bottlenecks that should be measured:** `RegionPredictor.predict_many` performs sequential single-item predictions instead of batching (`inference.py:601-608`); repeated filename/content fingerprinting can be expensive for very large track sets; header QC scans a summary bin for each chromosome. Add representative benchmarks before changing these. A batched `predict_many` is likely worthwhile because the model already supports batches.

**Not a target:** the chromosome-pass writer is a credible domain-specific optimisation and already has parity tests. Do not replace it with generic DataFrame code. Do not optimise tiny configuration or CLI paths.

### Security and privacy

No `shell=True`, `eval`, unsafe YAML loader, embedded credential, or obvious command-string injection was found. Subprocesses generally use argument arrays, and Snakemake shell inputs frequently use its quoting modifier. Release publishing appropriately scopes OIDC `id-token: write` to the publish job.

Confirmed or plausible risks:

- F2 is a destructive path-validation issue.
- F5 uses predictable shared temporary directories.
- F11 lacks safe output-name enforcement and may permit path traversal from crafted metadata.
- The legacy `.bin` warm-start path calls `torch.load(weight_path, map_location="cpu")` without explicitly setting `weights_only=True` (`training/runner.py:105-123`). The declared floor is Torch 2.5, whose default is not the restricted loader; Torch changed that default in 2.6. A checkpoint is therefore a potentially executable pickle for a supported environment. Pass `weights_only=True` explicitly, prefer the already-supported Safetensors path, and give a clear trust warning if a genuinely incompatible legacy file requires an explicit unsafe escape hatch. PyTorch warns never to load untrusted data and documents the restricted loader; Safetensors is specifically designed to store tensors without pickle execution: [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html), [PyTorch serialization security](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true), [Safetensors](https://huggingface.co/docs/safetensors/index).
- Training defaults save the complete dirty Git diff (`training/config.py` and `training/provenance.py`). Diffs can contain secrets or large proprietary data. Preserve commit hash and dirty status by default; make full diff capture opt-in, size-limited, and visibly warned.
- Dependency vulnerabilities were not established by this audit because a clean current environment cannot resolve. Add scanners after F4; do not claim a clean vulnerability state from the existing venv.

There is no custom HTTP client, authentication, retry, or connection-pooling layer to replace. Network access is delegated to Hugging Face `from_pretrained` calls and optional W&B integration. That is the correct boundary. Document which commands may download or upload data, expose the established offline/cache controls where relevant, and avoid building a wrapper client without a demonstrated policy requirement.

Use [pip-audit](https://pypa.github.io/pip-audit/) for Python environment/lock scanning and [cargo-audit/RustSec](https://rustsec.org/) for `Cargo.lock`. Both report known advisories; neither proves an application is secure. Pin scanner action versions or install versions deliberately, and define how exceptions expire.

### Developer experience and CI/CD

CI covers Ruff, Clippy, Python 3.12/3.13 data tests, a Python 3.12 model job, and schema drift. Release builds four platform architectures for both supported Python versions and uses trusted publishing. These are strong foundations.

Add, in order:

1. `uv lock --check` after fixing F4.
2. `cargo fmt --check` and `cargo test --all-targets` (the latter becomes valuable after Rust tests exist).
3. Build each wheel/sdist and install it into a clean environment; run import and CLI smoke tests.
4. Run the entire Python suite in at least one job, so new test files are not silently omitted from hard-coded lists.
5. `pip-audit` and `cargo audit`, initially non-blocking while triaging the baseline, then blocking with documented exceptions.
6. Correct the model-job comment that says “four files” while invoking eight.

Do not add a large matrix of every dependency version. One current environment plus a periodic lowest-direct-bound environment is sufficient if lower bounds are claimed. Unbounded dependencies such as `polars`, `pyarrow`, `typer`, and several extras should either receive tested lower bounds or be documented as intentionally floating during alpha.

## Library and standard-library review

The audit found more opportunities to remove dependencies than to add them. No new runtime library is necessary for the main simplifications.

| Current implementation | Purpose | Recommendation | Library / stdlib option | Why | Migration complexity | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed scratch paths and manual cleanup | Build staging | Replace | `tempfile.TemporaryDirectory` | Unique invocation scope and reliable cleanup with no dependency | Low | `dataset/build.py:496-505`; [tempfile](https://docs.python.org/3/library/tempfile.html) |
| Direct multi-file output mutation | Dataset/recompression publication | Simplify with one helper | sibling temp directory + `os.replace` | Makes validation-before-publication explicit; standard library is sufficient | Medium | F1/F2; [os.replace](https://docs.python.org/3/library/os.html#os.replace) |
| `list(reader)` | Arrow recompression | Replace | Arrow IPC batch iteration | Same Arrow dependency, bounded memory, simpler loop | Low | `recompress.py:25-29`; [Arrow IPC](https://arrow.apache.org/docs/python/ipc.html) |
| Bioframe `read_table(schema="bed")` | Read three BED columns | Replace and remove dependency | existing `pandas.read_csv` | pandas is already core and supplies needed parsing controls | Low | `normalization.py:20-43`; [read_csv](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html) |
| Loguru plus stdlib logging | Application logs | Standardise and remove Loguru | `logging.getLogger(__name__)` | Current code uses no Loguru feature that justifies a second system; library/application integration is clearer | Medium | 13 Loguru-using modules plus stdlib in inference/normalisation; [logging](https://docs.python.org/3/library/logging.html) |
| Generic `PyRuntimeError` at native boundary | Python error taxonomy | Simplify | PyO3 built-in exceptions; at most one custom data error | Stable categories without a framework or deep hierarchy | Low/medium | Rust PyO3 entry points; [PyO3 exceptions](https://pyo3.rs/main/exception) |
| Manual workflow cross-field checks | Config semantics | Consolidate, do not add library | existing Pydantic model; generated JSON Schema | One semantic source already exists | Medium | `config/models.py`, Snakefile; [Pydantic schema](https://docs.pydantic.dev/latest/concepts/json_schema/) |
| `polars` direct dependency | No identified runtime purpose | Remove | none | Zero imports; pandas/Arrow already cover actual paths | Low | repository import search; [Polars registry](https://pypi.org/project/polars/) |
| `pysam`, `psutil` in data extra | No identified product import | Remove from data | none; keep psutil only as FlashAttention build dependency if required | Avoid misleading heavy/platform dependencies | Low/medium | repository import search; existing uv build-dependency declaration |
| `tangermeme` | Attribution, nominally | Remove | retain current bounded custom algorithm | The code explicitly avoids Tangermeme because its function materialises mutants; unused package brings a heavy ML/scientific tree | Low | `design/attribution.py:145-157`, `docs/attribution.md:169-176`; [Tangermeme](https://pypi.org/project/tangermeme/) |
| `lightning`, `peft` direct train deps | No identified product import | Remove unless clean-install tests prove an indirect runtime need | Hugging Face Trainer / current custom transfer code | Direct dependencies should describe direct contracts, not possible transitive needs | Medium | import search and training implementation |
| `plotly`, `seaborn` | Optional visualisation | Remove unless an owned example imports them | existing matplotlib where callback needs it | Empty extras create maintenance without functionality | Low | no package/example imports found |
| Bigtools Rust + pybigtools Python backends | BigWig access | **Keep** | current bigtools ecosystem | Appropriate native performance and a Python API for ancillary scans; duplication has distinct roles | None | native writer, QC, inference; [bigtools](https://github.com/jackh726/bigtools) |
| Typer / Hydra / Snakemake | CLI / config composition / DAG execution | **Keep all three** | current libraries | They solve distinct problems. Removing one would push complexity into custom code | None | CLI, `python/configs`, packaged workflow |
| Pandera track-table validation | DataFrame schema | Keep for now | alternative: bespoke pandas checks | Cross-row/table validation is useful; removal would recreate schema logic. Reassess only if core installation size is a measured problem | None | `tracks_table.py` and tests |

### Dependency inventory decision summary

| Dependency/group | Decision | Rationale |
| --- | --- | --- |
| NumPy, pandas, PyArrow | Keep | Central array/table/Arrow representation |
| Pydantic | Keep | Canonical workflow configuration and schema generation |
| Pandera | Keep provisionally | Canonical track-table validation has real table-level constraints |
| tqdm | Keep | Used for long direct operations; small and appropriate |
| Typer | Keep | CLI help/typing is substantially better than bespoke argparse dispatch here |
| datasets | Keep in data | On-disk Dataset/DatasetDict compatibility is an intentional external contract |
| pybigtools | Keep in data | Direct QC/inference BigWig API; shares underlying ecosystem with Rust core |
| SciPy | Verify then keep/remove | Normalisation likely uses statistical functions; retain only if import search confirms it after current refactors |
| Torch, Transformers, Accelerate, TorchMetrics, backbone packages, Hydra, W&B, Matplotlib | Keep where used | Actual training/model/callback contracts; extras appropriately isolate weight |
| Snakemake | Keep in workflow | Correct restartable cluster-DAG abstraction |
| FlashAttention/Ninja | Keep GPU-only | Correctly isolated specialised optional path |
| Polars, Bioframe, Tangermeme, Pysam, Psutil(data), Lightning, PEFT, Plotly, Seaborn | Remove or prove use | No direct production contract, or trivially replaceable single use |
| Loguru | Remove after migration | Redundant logging stack, not abandoned software |
| SeqNado | Correct metadata | Valuable optional integration, but declared version does not exist |

## Target architecture and the most important simplifications

The target remains a modular monolith with one native extension. It does not need services, plugins, dependency injection, or a generic pipeline framework.

```text
CLI adapters (Typer)                    Python facade
        |                                    |
        +------------- application services-+
                         |
       +-----------------+-------------------------------+
       |                 |               |               |
   tracks service   dataset service  training service  prediction/design
       |                 |               |               |
 Pydantic/Pandera    native writer   Hydra-resolved    model/checkpoint
 validation           + Arrow         typed config       contract
       |                 |
       +------ canonical files --------+
              tracks.parquet / dataset / checkpoints

Snakemake calls the same services through thin CLI commands and owns only DAG concerns.
```

The ten highest-value architectural simplifications are:

1. Establish one transactional directory-publication primitive and use it in build/recompress.
2. Give each file-producing invocation private staging and explicit resource ownership.
3. Make Pydantic the only workflow semantic validator; generate structural schema from it.
4. Validate the entire Hydra training configuration before any expensive work.
5. Turn CLI functions into adapters around domain services; split registration by command group.
6. Expose a five-to-eight-name Python facade; treat implementation modules as advanced/unstable.
7. Delete legacy Python signal readers and no-op build controls.
8. Standardise logging on the standard library at module boundaries.
9. Shrink extras to dependencies exercised by a clean-install test.
10. Stream Arrow and prediction outputs instead of collecting workload-sized lists.

## Ranked issue list

| Rank | Issue | Priority | Impact | Effort | Change risk | Compatibility |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Preserve skipped splits during incremental publication | P0 | Prevents data loss | M | M | No format/API break |
| 2 | Reject destructive recompression paths; publish transactionally | P0 | Prevents data loss/partial state | M | M | Invalid invocations now error |
| 3 | Propagate Rust BigWig errors instead of zeros | P0 | Prevents silent scientific corruption | M | M | Failed reads now abort |
| 4 | Repair SeqNado metadata and regenerate uv lock | P0/P1 | Restores reproducible setup/release | S–M | L | Packaging change |
| 5 | Use invocation-private scratch directories | P1 | Eliminates build races | S | L | Internal only |
| 6 | Reject unknown QC rules | P1 | Restores QC contract | S | L | Invalid configs now error |
| 7 | Stream recompression batches | P1 | Bounds memory | S | L | Internal only |
| 8 | Consolidate workflow validation; add anchor generator support | P1 | Predictable configuration | M | M | Earlier failures |
| 9 | Remove no-op build controls and rename API with alias | P1 | Honest, smaller API | M | L | Deprecation cycle |
| 10 | Validate a complete typed training config at entry | P1 | Earlier, clearer failures | M–L | M | Mapping adapter retained |
| 11 | Remove unused/heavy direct dependencies | P1 | Faster, more reliable installs | M | M | Extra contents change |
| 12 | Bound genome-wide prediction output memory | P1/P2 | Makes advertised workload reliable | M–L | M | Same output contract |
| 13 | Sanitize prediction filenames | P2 | Prevents unsafe paths/collisions | S | L | Filenames may change; mapping added |
| 14 | Split giant CLI module around existing services | P2 | Testability and navigation | L | M | Commands preserved |
| 15 | Delete dead dataset reader path | P2 | ~175 lines removed | S | L | Private code only unless accidental users |
| 16 | Replace wildcard dataset/scaling facades | P2/P3 | Explicit API boundary | S–M | L | Deprecate accidental exports |
| 17 | Standardise stdlib logging | P2 | Consistent embedding/configuration | M | L | Log format may change |
| 18 | Add native/failure-path/clean-install CI | P1/P2 | Protects critical contracts | M | L | No runtime break |
| 19 | Make full Git diff provenance opt-in | P2 | Reduces secret/data leakage | S | L | Provenance content changes |
| 20 | Restrict legacy `torch.load` and prefer Safetensors | P1 | Prevents checkpoint pickle execution on supported Torch 2.5 | S | L | Exotic legacy state dicts may need an explicit trusted escape hatch |
| 21 | Batch `predict_many` after benchmark | P3 | Improves interactive throughput | M | L | Same result type |

## Refactoring roadmap

### Phase 0 — Protect existing behaviour and data

**Work:** Add regression tests for F1–F3, recompression path relationships/interruption, QC unknown rules, and concurrent scratch. Record golden dataset metadata and prediction intervals. Add full-suite CI and fix the Ruff import order.

**Dependencies:** None. Do this before implementation changes.  
**Benefit:** Converts the highest-risk implicit contracts into executable ones.  
**Risk / complexity:** Low risk, medium effort.  
**Incremental:** Yes; no public behaviour changes.

### Phase 1 — Make writes safe and environments reproducible

**Work:** Implement private scratch and transactional per-split publication; make recompression validate paths and stream; propagate native read errors; fix SeqNado metadata and regenerate the lock; add lock/fmt/native-test checks to CI.

**Dependencies:** Phase 0 tests. Transactional directory helper should land before both callers migrate.  
**Benefit:** Removes destructive behaviour, silent corruption, memory spikes, races, and broken onboarding.  
**Risk / complexity:** Medium risk, medium-to-large combined effort.  
**Incremental:** Yes. On-disk formats stay unchanged. Invalid/silent cases intentionally become explicit errors.

### Phase 2 — Low-risk contraction

**Work:** Delete dead dataset helpers; deprecate no-op parameters and `build_dataset_fast`; replace Bioframe parsing; remove unused dependencies in small groups; replace wildcard facades with explicit exports; fix README/config documentation contradictions; make Git diff provenance opt-in.

**Dependencies:** Clean-install CI from Phase 1 makes dependency removal safe.  
**Benefit:** Smaller API, source, install, and documentation surface.  
**Risk / complexity:** Low-to-medium.  
**Incremental:** Yes. Use one release of warnings for public-looking aliases/options.

### Phase 3 — Establish typed service boundaries

**Work:** Validate the complete training config at Hydra entry; consolidate Pydantic/Snakemake semantics; add anchor generator path; define `DatasetBuildResult` and `TrainingSummary`; create the slim top-level facade; reject unknown QC metrics and validate BED rows.

**Dependencies:** Dataset safety work should settle before crystallising its public result.  
**Benefit:** Earlier failures, discoverable common workflows, simpler tests, one configuration truth.  
**Risk / complexity:** Medium.  
**Incremental:** Mapping and import aliases preserve compatibility while callers move.

### Phase 4 — Decompose orchestration without changing commands

**Work:** Extract design, attribution, build, prediction, and training services; split Typer command registration modules; divide `run_training` into dataset/model/trainer/execution functions; extract TrackSheet input adapters. Standardise logging during these moves to avoid a standalone churn-only rewrite.

**Dependencies:** Typed boundaries from Phase 3.  
**Benefit:** Large reduction in cognitive load and much easier service-level tests.  
**Risk / complexity:** Medium-to-high risk, large effort.  
**Incremental:** One command/service at a time. Snapshot CLI help and run golden end-to-end tests. No command rename is required.

### Phase 5 — Bounded inference and polish

**Work:** Spool/stream genome-wide outputs; sanitize filenames with a manifest; batch `predict_many` after benchmarking; complete Python API, troubleshooting, and architecture docs; add periodic dependency/advisory checks.

**Dependencies:** Stable service and resource-lifetime boundaries.  
**Benefit:** Reliable genome-scale operation and a clearer new-user experience.  
**Risk / complexity:** Medium, primarily around byte-equivalent BigWig output.  
**Incremental:** Yes. Regional prediction can remain untouched while the genome-wide writer changes.

## Quick wins

These can become small, reviewable tickets immediately:

1. Fix the Ruff `I001` in `design/diagnostics.py`.
2. Change the README sentence that says the workflow does not perform QC.
3. Make unknown QC rule columns an error and reverse the existing “ignores” test.
4. Add `anchor` to generator choices with its required region prompts.
5. Fix the multi-project SeqNado Pydantic condition to match its comment and Snakefile.
6. Add `cargo fmt --check`, `cargo test --all-targets`, and full `pytest tests` to CI.
7. Remove `polars` after a clean core import test.
8. Replace the single Bioframe reader with pandas, then remove Bioframe.
9. Remove Tangermeme after verifying no owned notebook imports it; correct its documentation claim.
10. Delete the four unreferenced dataset helpers and their now-unused controls in a dedicated commit.
11. Replace the fixed `tracks.parquet.tmp` name with a random same-directory temporary file.
12. Put a size limit and explicit opt-in around captured Git diffs.
13. Pass `weights_only=True` explicitly to the legacy `.bin` checkpoint loader and add an untrusted-checkpoint regression test.

Do not label the data-loss fixes “quick wins” even if their patches are short; their validation burden is significant.

## High-impact changes

- Transactional dataset/recompression publication.
- Strict native data-read error propagation.
- Reproducible packaging and clean-install matrix.
- Complete typed training configuration at the actual boundary.
- Bounded-memory genome-wide prediction.
- Service extraction from the giant CLI and runner, after behavioural tests exist.

These changes materially improve correctness and maintainability. They should be reviewed as invariant-preserving engineering work, not bundled with cosmetic formatting.

## Things to leave alone

1. **The Rust/Python split.** BigWig binning and Arrow writing are credible native hot paths; Python is the right orchestration layer.
2. **The chromosome-pass strategy.** It addresses the dominant repeated-seek workload and has parity coverage. Benchmark it, but do not replace it with generic code.
3. **The canonical `tracks.parquet` direction.** One table carrying track identity, status, fingerprints, scale, QC, and later metadata is a strong simplification.
4. **Typer, Hydra, and Snakemake together.** They are not three competing frameworks: one parses commands, one composes training presets, and one schedules a restartable DAG.
5. **Pydantic-generated workflow schema.** This is the correct move toward a single source of truth; finish the consolidation rather than undoing it.
6. **Lazy top-level model imports.** They let core installations import without Torch. Keep laziness while shrinking the export list.
7. **Self-contained modern checkpoints.** The checkpoint/config/weights contract is valuable. Preserve legacy loading as a clearly bounded compatibility path.
8. **Behavioural writer parity tests.** They are high value; supplement them with independent failure tests rather than replacing them.
9. **Explicit optional extras.** Data, training, GPU, workflow, visualisation, and notebooks are reasonable installation boundaries; populate them honestly.
10. **Domain-specific attribution implementation.** The comment gives a real memory reason not to delegate to Tangermeme. Remove the unused dependency, not the bounded implementation.
11. **No async conversion.** The workload is CPU/I/O batch processing and native parallelism, not a network service. Async would add concepts without solving the identified races.
12. **No plugin/microservice architecture.** There is no evidence that deployment or team boundaries require it.

## Overall assessment

| Dimension | Rating | Explanation |
| --- | ---: | --- |
| Maintainability | **6/10** | Clear domain modules and good tests are offset by concentrated 700–2,400-line modules, accidental exports, and orchestration-heavy functions. |
| Complexity | **6/10** | Most domain complexity is legitimate, but dead paths, no-op parameters, duplicate validation, and oversized CLI/training functions add avoidable cognitive load. Higher is better here. |
| API quality | **5/10** | CLI discovery is decent; the Python facade exposes internals instead of common workflows, and several accepted controls do nothing. |
| Robustness | **4/10** | Happy paths pass, but destructive publication, self-deleting recompression, silent read failures, and shared scratch are serious operational gaps. |
| Extensibility | **7/10** | New tracks, models, heads, presets, and workflow stages have plausible homes. Large command/runner functions make extensions more expensive than necessary. |
| Dependency hygiene | **4/10** | Extras are conceptually separated, but several direct packages are unused, one requirement is impossible, and the lock is stale. |
| Documentation quality | **7/10** | There are substantial task-oriented guides and useful examples. Contradictions, broken setup, and absent API/failure contracts prevent a higher score. |
| Testing quality | **7/10** | Broad, fast behavioural tests and good domain coverage. Critical failure paths and native unit tests are missing, and current CI does not run every test. |

**Overall: 6/10.** This is a functional, promising scientific codebase with an appropriate basic architecture. It is not in need of reinvention. It is in need of stronger durability invariants, honest dependency/API contracts, and deliberate deletion. Completing Phases 0–2 would remove the most dangerous behaviour and a meaningful amount of incidental surface without destabilising the scientific core. Phases 3–4 would then make the code dramatically easier for another engineer to understand and extend.

## External sources consulted

All external links below were checked for the ecosystem guidance used in this audit; package-version observations are dated 2026-09-10.

- [uv universal resolution](https://docs.astral.sh/uv/concepts/resolution/)
- [uv project configuration](https://docs.astral.sh/uv/concepts/projects/config/)
- [PyPA dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)
- [SeqNado on PyPI](https://pypi.org/project/seqnado/)
- [Python `tempfile`](https://docs.python.org/3/library/tempfile.html)
- [Python `os.replace`](https://docs.python.org/3/library/os.html#os.replace)
- [Python `shutil`](https://docs.python.org/3/library/shutil.html)
- [Python logging](https://docs.python.org/3/library/logging.html)
- [Apache Arrow IPC](https://arrow.apache.org/docs/python/ipc.html)
- [pandas `read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
- [Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)
- [Snakemake configuration validation](https://snakemake.readthedocs.io/en/stable/snakefiles/configuration.html#validation)
- [PyO3 exception handling](https://pyo3.rs/main/exception)
- [bigtools / pybigtools](https://github.com/jackh726/bigtools)
- [Hypothesis documentation](https://hypothesis.readthedocs.io/)
- [pip-audit](https://pypa.github.io/pip-audit/)
- [RustSec](https://rustsec.org/)
- [Polars on PyPI](https://pypi.org/project/polars/)
- [Tangermeme on PyPI](https://pypi.org/project/tangermeme/)
- [Bioframe on PyPI](https://pypi.org/project/bioframe/)
- [Loguru repository and releases](https://github.com/Delgan/loguru)
- [PyTorch `torch.load`](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
- [PyTorch serialization security](https://docs.pytorch.org/docs/stable/notes/serialization.html#torch-load-with-weights-only-true)
- [Safetensors](https://huggingface.co/docs/safetensors/index)
