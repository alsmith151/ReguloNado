TMM golden fixture (audit F05 / R3 / F19-partial)
==================================================

Files
-----
counts.csv              200 regions x 4 tracks of pseudo-counts (float, includes
                         exact zeros). Column names track0..track3.
library_sizes.csv        Per-track whole-library mapped-read counts (1 column,
                         4 rows), deliberately NOT proportional to
                         counts.csv's column sums -- this mirrors regulonado's
                         real usage where the TMM M/A comparison runs over a
                         curated region subset (counts.csv) but scaling uses
                         the track's true total library size
                         (library_sizes.csv), i.e. an explicit edgeR
                         `lib.size=` rather than `colSums(counts)`.
edger_norm_factors.csv   edgeR::calcNormFactors(..., method="TMM") output for
                         counts.csv with the explicit library_sizes.csv above.
                         This is the ground truth compute_tmm_factors /
                         _tmm_from_counts must reproduce.

Fixture generation (Python, numpy, seed=42)
--------------------------------------------
    rng = np.random.default_rng(42)
    n_regions, n_tracks = 200, 4
    region_length_kb = 200.0
    library_sizes = np.array([12_000_000., 30_000_000., 8_000_000., 20_000_000.])
    baseline = rng.gamma(shape=2.0, scale=300.0, size=n_regions)
    sample_scale = np.array([1.0, 1.8, 0.6, 2.5])
    counts = baseline[:, None] * sample_scale[None, :]
    hot_idx = rng.choice(n_regions, size=8, replace=False)
    counts[hot_idx, 1] *= 40.0          # composition-bias outliers in one track
    counts = rng.poisson(counts).astype(float)
    zero_mask = rng.random(counts.shape) < 0.08
    counts[zero_mask] = 0.0             # ~8% zero-inflation

Counts are kept in the hundreds-to-thousands range (gamma scale=300) rather
than tens, because edgeR's TMM trims by *rank*, and small integer counts
produce many exact ties in the M statistic; ties make rank-based trimming
sensitive to which implementation's tie-breaking runs first, which would make
the golden comparison noisy for reasons unrelated to algorithmic correctness.

Ground truth generation (R / edgeR, exact command used, run once via pixi;
edgeR is a fixture-generation tool only, never a runtime or test dependency)
-----------------------------------------------------------------------------
    pixi exec -s r-base -s bioconductor-edger -c conda-forge -c bioconda -- \
        Rscript -e '
            library(edgeR)
            counts <- as.matrix(read.csv("counts.csv"))
            lib <- read.csv("library_sizes.csv")$library_size
            nf <- calcNormFactors(counts, lib.size = lib, method = "TMM",
                                   logratioTrim = 0.3, sumTrim = 0.05,
                                   doWeighting = TRUE)
            write.csv(data.frame(track = colnames(counts), tmm_factor = nf),
                      "edger_norm_factors.csv", row.names = FALSE)
        '
    # edgeR version used: 4.8.2 (bioconductor, installed via conda-forge/bioconda)

Why not rnanorm (see final report / commit message for full reasoning)
------------------------------------------------------------------------
rnanorm.TMM always derives each sample's "library size" internally as
colSums of the count matrix passed to it -- it has no parameter to accept an
externally supplied library size. Feeding it counts.csv directly would
silently substitute colSums(counts.csv) for library_sizes.csv, which is
exactly the behaviour this fixture is designed to rule out (see
library_sizes.csv note above). Empirically, on this fixture, rnanorm's
output differs from the correct (explicit lib.size) edgeR ground truth by up
to 0.40 in factor units (e.g. track3: 1.50 vs rnanorm's 1.12) -- while
matching edgeR run with lib.size=colSums(counts) to 1e-15, exactly as
rnanorm's own docs claim. rnanorm was correspondingly not adopted; see
compute_tmm_factors's docstring.
