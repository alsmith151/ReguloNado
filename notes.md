
# Fine-tuning notes template

Replace this file with notes specific to your fine-tuning experiment.

## Dataset

Describe the genomic tracks used for training:
- Assay types (e.g. ChIP-seq, ATAC-seq, RNA-seq, CAGE-seq)
- Cell lines or conditions
- Number of tracks and samples

## Model

Note the backbone used (e.g. Borzoi pretrained on human/mouse multi-assay tracks) and
which head architecture was selected. Describe any pretrained heads that were replaced.

## Training

Summarise the training run:
- Hardware (e.g. 2× L40 GPUs)
- Batch size, learning rate, number of steps
- Which Borzoi genomic regions / folds were used for train/validation/test

## Analysis

Describe any downstream analyses (e.g. saliency maps, in-silico perturbations, locus
visualisations) and the genomic coordinates of interest.
