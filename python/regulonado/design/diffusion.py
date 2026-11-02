"""Small numeric-conditioned masked diffusion model for regulatory DNA.

Activity columns have explicit names and assay units. Standardisation is learned
on training chromosomes only and travels with the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from regulonado.config.diffusion import DiffusionConfig

MASK, PAD, UNKNOWN = 4, 5, 6


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode(sequence: str):
    return torch.tensor(
        ["ACGT".find(base) if base in "ACGT" else UNKNOWN for base in sequence.upper()],
        dtype=torch.long,
    )


class Denoiser(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        if config.architecture == "alphagenome_encoder":
            if not config.encoder_checkpoint:
                raise ValueError("encoder_checkpoint is required for encoder diffusion")
            from regulonado.design.encoder_diffusion import EncoderDenoiser

            self.network = EncoderDenoiser(config)
            return
        self.tokens = nn.Embedding(7, config.width, padding_idx=PAD)
        self.positions = nn.Embedding(config.max_length, config.width)
        self.condition = nn.Sequential(
            nn.Linear(2 * len(config.activity_columns), config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.time = nn.Sequential(
            nn.Linear(1, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        layer = nn.TransformerEncoderLayer(
            config.width,
            config.heads,
            config.width * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.output = nn.Linear(config.width, 4)

    def forward(self, tokens, time, values, observed):
        if self.config.architecture == "alphagenome_encoder":
            return self.network(tokens, time, values, observed)
        length = tokens.shape[1]
        if length > self.config.max_length:
            raise ValueError("Sequence exceeds denoiser max_length")
        conditioning = torch.cat([values.masked_fill(~observed, 0), observed.float()], dim=-1)
        hidden = self.tokens(tokens) + self.positions(torch.arange(length, device=tokens.device))
        hidden = hidden + self.condition(conditioning)[:, None] + self.time(time[:, None])[:, None]
        return self.output(self.encoder(hidden, src_key_padding_mask=tokens.eq(PAD)))


def mdlm_loss(model, clean, values, observed, *, condition_dropout=0.0, generator=None):
    """Monte Carlo MDLM bound, alpha(t)=1-t; zero loss on unmasked bases.

    Normalise by all valid bases, *not* by the sampled number of masks.
    """
    draws = torch.rand(clean.shape[0], generator=generator)
    if getattr(getattr(model, "config", None), "stratified_timesteps", False):
        draws = (draws + torch.arange(len(clean))) / len(clean)
        draws = draws[torch.randperm(len(clean), generator=generator)]
    times = draws.to(clean.device).clamp_min(1e-4)
    valid = clean.lt(4)
    masked = (
        torch.rand(clean.shape, generator=generator).to(clean.device) < times[:, None]
    ) & valid
    noisy = clean.masked_fill(masked, MASK)
    keep = torch.rand(clean.shape[0], generator=generator).to(clean.device) >= condition_dropout
    logits = model(noisy, times, values, observed & keep[:, None])
    loss = F.cross_entropy(logits.transpose(1, 2), clean.clamp_max(3), reduction="none")
    return ((loss * masked / times[:, None]).sum(1) / valid.sum(1).clamp_min(1)).mean()


@torch.no_grad()
def denoising_diagnostics(model, clean, values, observed, *, prefix, seed):
    """Fixed-mask CE (nats/masked base) and accuracy, without diffusion weighting."""
    valid = clean.lt(4)
    metrics = {}
    for rate, label in [(0.15, "15pct"), (0.5, "50pct"), (0.9, "90pct")]:
        generator = torch.Generator().manual_seed(seed + int(rate * 100))
        masked = (torch.rand(clean.shape, generator=generator).to(clean.device) < rate) & valid
        logits = model(
            clean.masked_fill(masked, MASK),
            torch.full((len(clean),), rate, device=clean.device),
            values,
            observed,
        )
        per_token = F.cross_entropy(logits.transpose(1, 2), clean.clamp_max(3), reduction="none")
        denominator = masked.sum().clamp_min(1)
        metrics[f"{prefix}_masked_nll_{label}"] = float((per_token * masked).sum() / denominator)
        correct = logits.argmax(-1).eq(clean) & masked
        metrics[f"{prefix}_masked_accuracy_{label}"] = float(correct.sum() / denominator)
    return metrics


@torch.no_grad()
def sample(model, values, observed, *, length, count=16, steps=64, guidance=1.0, seed=0):
    """Absorbing-mask reverse process. Revealed bases are copied exactly."""
    if not 1 <= length <= model.config.max_length or count < 1 or steps < 1:
        raise ValueError("Invalid generation length, count, or steps")
    if not np.isfinite(guidance) or guidance < 0:
        raise ValueError("guidance must be finite and nonnegative")
    device = next(model.parameters()).device
    # CPU RNG provides repeatable draws without requiring MPS generator support.
    generator = torch.Generator().manual_seed(seed)
    values = torch.as_tensor(values, dtype=torch.float32, device=device).expand(count, -1)
    observed = torch.as_tensor(observed, dtype=torch.bool, device=device).expand(count, -1)
    tokens = torch.full((count, length), MASK, dtype=torch.long, device=device)
    model.eval()
    for index in range(steps, 0, -1):
        times = torch.full((count,), index / steps, device=device)
        logits = model(tokens, times, values, observed)
        if guidance != 1:
            null = model(tokens, times, values, torch.zeros_like(observed))
            logits = null + guidance * (logits - null)
        probabilities = logits.softmax(-1).cpu().reshape(-1, 4)
        bases = (
            torch.multinomial(probabilities, 1, generator=generator)
            .reshape(count, length)
            .to(device)
        )
        # (alpha(s)-alpha(t))/(1-alpha(t)) = (t-s)/t = 1/index.
        reveal = (torch.rand((count, length), generator=generator) < 1 / index).to(device)
        tokens = torch.where(tokens.eq(MASK) & reveal, bases, tokens)
    return ["".join("ACGT"[base] for base in row) for row in tokens.cpu().tolist()]


def load_checkpoint(path, device="auto"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("Unsupported diffusion checkpoint format")
    model = Denoiser(DiffusionConfig.model_validate(checkpoint["config"]))
    if checkpoint["format_version"] == 2:
        if model.config.architecture != "alphagenome_encoder":
            raise ValueError("Encoder decoder checkpoint requires encoder architecture")
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        if unexpected or any(not name.startswith("network.backbone.") for name in missing):
            raise ValueError("Encoder decoder checkpoint is missing trainable weights")
    else:
        model.load_state_dict(checkpoint["model"])
    return model.to(device_for(device)).eval(), checkpoint


def condition_vector(checkpoint, activities):
    names = checkpoint["config"]["activity_columns"]
    unknown = set(activities) - set(names)
    if unknown:
        raise ValueError(f"Unknown activity dimensions: {sorted(unknown)}")
    raw = np.array([activities.get(name, np.nan) for name in names], dtype=float)
    if np.isinf(raw).any():
        raise ValueError("Activity values must be finite or missing")
    observed = np.isfinite(raw)
    values = np.where(observed, (raw - checkpoint["mean"]) / checkpoint["scale"], 0)
    return values, observed


def train_diffusion(table_path, out_dir, config: DiffusionConfig, *, resume=None, on_epoch=None):
    """Train from TSV/parquet: chrom, sequence, and named numeric activities.

    NaN activity is unobserved, not zero. Sequences may alternatively be prepared
    from BED/FASTA upstream; the explicit table keeps assay transformations auditable.
    """
    import hashlib
    import json
    import logging

    import pandas as pd
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    if config.architecture == "alphagenome_encoder":
        if not config.encoder_checkpoint:
            raise ValueError("encoder_checkpoint is required for encoder diffusion")
        from regulonado.design.encoder_diffusion import encoder_digest

        config = config.model_copy(
            update={"encoder_sha256": encoder_digest(config.encoder_checkpoint)}
        )
    torch.manual_seed(config.seed)
    table_path, out_dir = Path(table_path), Path(out_dir)
    with table_path.open("rb") as handle:
        table_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    table = (
        pd.read_parquet(table_path)
        if table_path.suffix == ".parquet"
        else pd.read_csv(table_path, sep="\t")
    )
    required = ["chrom", "sequence", *config.activity_columns]
    if any(name not in table for name in required):
        raise ValueError(f"Training table requires columns {required}")
    if table["chrom"].isna().any() or table["sequence"].isna().any():
        raise ValueError("chrom and sequence must not be missing")
    sequences = table.sequence.astype(str).str.upper().tolist()
    if any(
        not 1 <= len(s) <= config.max_length or set(s) - set("ACGTN") or not set(s) & set("ACGT")
        for s in sequences
    ):
        raise ValueError("Sequences must contain A/C/G/T/N, valid bases, and fit max_length")
    if table.chrom.astype(str).isin(config.heldout_chromosomes).any():
        raise ValueError("Held-out test chromosomes are present in the training table")
    if "split" in table:
        from regulonado.design.diffusion_data import chromosome_splits

        table = chromosome_splits(table, config.validation_chromosomes, config.heldout_chromosomes)
        if table.split.eq("test").any():
            raise ValueError("Held-out test rows are present in the training table")
        expected = table.chrom.astype(str).isin(config.validation_chromosomes)
        if not expected.equals(table.split.eq("validation")):
            raise ValueError("Recorded split labels disagree with validation chromosomes")
    validation = table.chrom.astype(str).isin(config.validation_chromosomes).to_numpy()
    if not validation.any() or validation.all():
        raise ValueError("Need nonempty training and validation chromosome splits")
    train_sequences = {s for s, heldout in zip(sequences, validation) if not heldout}
    train_sequences |= {
        s.translate(str.maketrans("ACGT", "TGCA"))[::-1] for s in train_sequences.copy()
    }
    if any(s in train_sequences for s, heldout in zip(sequences, validation) if heldout):
        raise ValueError(
            "Identical or reverse-complement sequences cross the train/validation split"
        )
    raw = table[config.activity_columns].to_numpy(dtype=float)
    if np.isinf(raw).any() or (~np.isfinite(raw[~validation])).all(axis=0).any():
        raise ValueError("Each activity needs finite training measurements; infinity is invalid")
    mean = np.nanmean(raw[~validation], axis=0)
    scale = np.nanstd(raw[~validation], axis=0)
    scale[scale < 1e-6] = 1
    observed = np.isfinite(raw)
    values = np.where(observed, (raw - mean) / scale, 0).astype(np.float32)
    tokens = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, : len(sequence)] = encode(sequence)

    def dataset(indices):
        return TensorDataset(
            tokens[indices], torch.from_numpy(values[indices]), torch.from_numpy(observed[indices])
        )

    training_data = dataset(~validation)
    sampler = None
    if config.condition_sampling_weights or config.sampling_weight_column:
        weights = np.ones(int((~validation).sum()), dtype=float)
        if config.sampling_weight_column:
            if config.sampling_weight_column not in table:
                raise ValueError("Missing sampling weight column")
            weights = table.loc[~validation, config.sampling_weight_column].to_numpy(dtype=float)
            if not np.isfinite(weights).all() or (weights <= 0).any():
                raise ValueError("Sampling weights must be finite and positive")
        for column, weight in config.condition_sampling_weights.items():
            weights = np.maximum(
                weights,
                np.where(observed[~validation, config.activity_columns.index(column)], weight, 1.0),
            )
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
    device = device_for(config.device)
    loader = DataLoader(
        training_data,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        generator=torch.Generator().manual_seed(config.seed),
    )
    val_loader = DataLoader(
        dataset(validation),
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
    )
    model = Denoiser(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.5,
            patience=config.scheduler_patience,
            threshold=config.early_stopping_min_delta,
            threshold_mode="abs",
            min_lr=config.min_learning_rate,
        )
        if config.lr_schedule == "plateau"
        else None
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    best, history = float("inf"), []
    patience_best, stale_epochs, best_epoch = float("inf"), 0, 0
    start_epoch, global_step = 0, 0
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if saved.get("training_table_sha256") != table_sha256 or "training_state" not in saved:
            raise ValueError("Resume needs matching training data and saved optimiser state")
        mutable = {
            "epochs",
            "device",
            "num_workers",
            "encoder_checkpoint",
            "early_stopping_patience",
            "early_stopping_min_epochs",
        }
        for key, value in config.model_dump().items():
            if key not in mutable and saved["config"].get(key) != value:
                raise ValueError(f"Resume configuration differs: {key}")
        missing, unexpected = model.load_state_dict(saved["model"], strict=False)
        if unexpected or any(not key.startswith("network.backbone.") for key in missing):
            raise ValueError("Resume checkpoint is missing model weights")
        state = saved["training_state"]
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        if config.epochs < start_epoch:
            raise ValueError("epochs is smaller than the resumed epoch")
        best, best_epoch = state["best"], state["best_epoch"]
        patience_best, stale_epochs = state["patience_best"], state["stale_epochs"]
        history, global_step = state["history"], state["global_step"]
        torch.set_rng_state(state["torch_rng"])
        loader.generator.set_state(state["loader_rng"])
        if sampler is not None:
            sampler.generator.set_state(state["sampler_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    warmup_steps = config.warmup_epochs * len(loader)
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_total = train_count = 0
        for clean, activity, known in loader:
            # Vectorised reverse complementation; leave padding fixed.
            lengths = clean.ne(PAD).sum(1)
            positions = torch.arange(clean.shape[1])[None]
            reverse = (lengths[:, None] - 1 - positions).clamp_min(0)
            flipped = clean.gather(1, reverse)
            flipped = torch.where(flipped.lt(4), 3 - flipped, flipped)
            use = (torch.rand(len(clean)) < config.reverse_complement_probability)[:, None] & (
                positions < lengths[:, None]
            )
            clean = torch.where(use, flipped, clean)
            clean, activity, known = [
                tensor.to(device, non_blocking=True) for tensor in (clean, activity, known)
            ]
            if global_step < warmup_steps:
                for group in optimizer.param_groups:
                    group["lr"] = config.learning_rate * (global_step + 1) / warmup_steps
            optimizer.zero_grad(set_to_none=True)
            loss = mdlm_loss(
                model, clean, activity, known, condition_dropout=config.condition_dropout
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            train_total += float(loss.detach()) * len(clean)
            train_count += len(clean)
        model.eval()
        total = count = 0
        # Fixed validation corruption draws make epoch comparisons meaningful.
        with torch.no_grad():
            validation_rng = torch.Generator().manual_seed(config.seed + 1)
            for batch in val_loader:
                clean, activity, known = [item.to(device) for item in batch]
                total += float(
                    mdlm_loss(model, clean, activity, known, generator=validation_rng)
                ) * len(clean)
                count += len(clean)
        metric = total / count
        if not np.isfinite(metric):
            raise ValueError("Nonfinite validation loss; check training settings and activities")
        logging.getLogger(__name__).info("MDLM epoch %d: validation loss %.5f", epoch + 1, metric)
        entry = dict(epoch=epoch + 1, train_loss=train_total / train_count, validation_loss=metric)
        if config.diagnostic_examples:
            for prefix, mask, offset in [
                ("train", ~validation, 1000),
                ("validation", validation, 2000),
            ]:
                indices = np.flatnonzero(mask)
                indices = np.random.default_rng(config.seed + offset).choice(
                    indices, size=min(config.diagnostic_examples, len(indices)), replace=False
                )
                diagnostics = dataset(indices)
                entry.update(
                    denoising_diagnostics(
                        model,
                        *[tensor.to(device) for tensor in diagnostics.tensors],
                        prefix=prefix,
                        seed=config.seed + offset,
                    )
                )
        entry["learning_rate"] = float(optimizer.param_groups[0]["lr"])
        history.append(entry)
        checkpoint = dict(
            format_version=2 if config.architecture == "alphagenome_encoder" else 1,
            config=config.model_dump(),
            model={
                key: value
                for key, value in model.state_dict().items()
                if not (
                    config.architecture == "alphagenome_encoder"
                    and key.startswith("network.backbone.")
                )
            },
            mean=mean.tolist(),
            scale=scale.tolist(),
            epoch=epoch + 1,
            validation_loss=metric,
            training_table=str(table_path.resolve()),
            training_table_sha256=table_sha256,
            training_rows=int((~validation).sum()),
            validation_rows=int(validation.sum()),
        )
        if metric < best:
            best, best_epoch = metric, epoch + 1
            torch.save(checkpoint, out_dir / "best.pt")
        if metric < patience_best - config.early_stopping_min_delta:
            patience_best, stale_epochs = metric, 0
        else:
            stale_epochs += 1
        if scheduler is not None and epoch + 1 >= config.warmup_epochs:
            scheduler.step(metric)
        checkpoint["training_state"] = dict(
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else None,
            epoch=epoch + 1,
            best=best,
            best_epoch=best_epoch,
            patience_best=patience_best,
            stale_epochs=stale_epochs,
            history=history,
            global_step=global_step,
            torch_rng=torch.get_rng_state(),
            loader_rng=loader.generator.get_state(),
            sampler_rng=sampler.generator.get_state() if sampler is not None else None,
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
        )
        temporary = out_dir / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        if on_epoch is not None:
            on_epoch(entry)
        if (
            config.early_stopping_patience
            and epoch + 1 >= config.early_stopping_min_epochs
            and stale_epochs >= config.early_stopping_patience
        ):
            logging.getLogger(__name__).info("Early stopping after %d epochs", epoch + 1)
            break
    (out_dir / "training_summary.json").write_text(
        json.dumps(
            dict(
                best_epoch=best_epoch,
                best_validation_loss=best,
                epochs_completed=len(history),
                early_stopped=len(history) < config.epochs,
            ),
            indent=2,
        )
    )
    return out_dir / "best.pt"
