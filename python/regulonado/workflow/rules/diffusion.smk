"""Optional count-conditioned diffusion training, using the normal GPU profile."""
import json

if DIFFUSION:
    DIFFUSION_DIR = RESULTS / 'diffusion'
    DIFFUSION_DATASET = Path(DIFFUSION.dataset_dir) if DIFFUSION.dataset_dir else RESULTS / 'region_counts' / 'dataset'
    DIFFUSION_ENCODER_RUN = (phase_run_dir(DIFFUSION.encoder_run, final_phase(DIFFUSION.encoder_run))
                             if DIFFUSION.encoder_run else None)
    DIFFUSION_ENCODER_INPUTS = ([str(DIFFUSION_ENCODER_RUN / 'trainer_state.json')]
        if DIFFUSION_ENCODER_RUN else
        [str(Path(DIFFUSION.training.encoder_checkpoint) / name) for name in ['config.json', 'model.safetensors']]
        if DIFFUSION.training.encoder_checkpoint else [])

    rule diffusion_prepare:
        input:
            dataset=[str(DIFFUSION_DATASET / name) for name in ['regions.parquet', 'counts.parquet', 'tracks.parquet']],
            fasta=config['inputs']['fasta'],
            priorities=[DIFFUSION.specificity_regions] if DIFFUSION.specificity_regions else []
        output:
            table=str(DIFFUSION_DIR / 'data' / 'training.parquet'),
            manifest=str(DIFFUSION_DIR / 'data' / 'manifest.json'),
            settings=str(DIFFUSION_DIR / 'settings.json')
        threads: DIFFUSION.cpus
        resources:
            mem_mb=scaled_mem_mb(DIFFUSION.mem_mb),
            runtime=DIFFUSION.runtime_minutes
        params:
            settings=DIFFUSION.model_dump_json(),
            dataset=str(DIFFUSION_DATASET),
            out=str(DIFFUSION_DIR / 'data')
        run:
            Path(output.settings).parent.mkdir(parents=True, exist_ok=True)
            Path(output.settings).write_text(params.settings)
            shell('regulonado generate prepare --dataset-dir {params.dataset:q} --fasta {input.fasta:q} --params {output.settings:q} --out {params.out:q}')

    rule diffusion_train:
        input:
            table=rules.diffusion_prepare.output.table,
            manifest=rules.diffusion_prepare.output.manifest,
            settings=rules.diffusion_prepare.output.settings,
            encoder=DIFFUSION_ENCODER_INPUTS
        output:
            best=str(DIFFUSION_DIR / 'model' / 'best.pt'),
            last=str(DIFFUSION_DIR / 'model' / 'last.pt'),
            summary=str(DIFFUSION_DIR / 'model' / 'training_summary.json')
        threads: DIFFUSION.cpus
        resources:
            mem_mb=scaled_mem_mb(DIFFUSION.mem_mb),
            runtime=DIFFUSION.runtime_minutes,
            gpu=1
        params:
            out=str(DIFFUSION_DIR / 'model'),
            encoder_arg=(' --encoder-run-dir ' + __import__('shlex').quote(str(DIFFUSION_ENCODER_RUN))) if DIFFUSION_ENCODER_RUN else ''
        shell:
            'regulonado generate fit --table {input.table:q} --manifest {input.manifest:q} --params {input.settings:q} --out {params.out:q}{params.encoder_arg}'
