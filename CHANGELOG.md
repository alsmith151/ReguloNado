# Changelog

All notable changes to ReguloNado are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> **Note:** After the first release, this file is managed automatically by
> [release-please](https://github.com/googleapis/release-please) based on
> [conventional commits](https://www.conventionalcommits.org/).

---

## [1.0.0](https://github.com/alsmith151/ReguloNado/compare/regulonado-v0.1.0...regulonado-v1.0.0) (2026-11-06)


### ⚠ BREAKING CHANGES

* **training:** imports, the Hydra config name and the preset group are renamed; no aliases are kept.
* **embeddings:** existing Parquet embedding caches are not read; re-embed them. `data.preload` is renamed `data.in_memory` and now reads the whole cache into RAM once; `embed regions --row-group-size` is removed.
* **config:** configure every model as a run with backbone, trunk, target and recipe
* **training:** train cached-trunk heads through `regulonado train`

### Features

* **adapter:** adapt Borzoi for transformers v5 weight loading and improve architecture building ([4a9fff2](https://github.com/alsmith151/ReguloNado/commit/4a9fff2d498c2811f3cad3ab99c33a498b89ec62))
* **adapter:** adapt Borzoi for transformers v5 weight loading and improve architecture building ([725e400](https://github.com/alsmith151/ReguloNado/commit/725e400c0cb6c698460a53d85cb820146f4bbc2d))
* add anchor-based normalization factors ([224af9c](https://github.com/alsmith151/ReguloNado/commit/224af9c60f764bf3cc2a2d3c3cb014d7197d68c4))
* add anchor-scaled target specificity score ([e1ea025](https://github.com/alsmith151/ReguloNado/commit/e1ea02536da7d5f42d918b9b0c95f33cf066968b))
* add SeqNado interoperability ([d23d3a4](https://github.com/alsmith151/ReguloNado/commit/d23d3a43642bfe97436d899ef1b467a8af7ae703))
* apply anchor backgrounds during training ([f9dc05a](https://github.com/alsmith151/ReguloNado/commit/f9dc05a3e3cdbd6d3bbc56489788c46452fa0230))
* **attribute:** add local rule for sharding attribution candidates ([4e7e9e9](https://github.com/alsmith151/ReguloNado/commit/4e7e9e9aeeff99cb43f8f1de1b4b27a7be6a49a3))
* **attribute:** remove topk_bins if bin_reduction is not set to topk ([6b6560f](https://github.com/alsmith151/ReguloNado/commit/6b6560ffe51dabe02f35db441eb103c97f34fb98))
* **attribute:** support calling multiple cores and score-only writes ([8571b5a](https://github.com/alsmith151/ReguloNado/commit/8571b5a25f107d0c0336a1e8b75a4b19c63ec685))
* **attribution:** add gradient-x-input as a fast ISM alternative ([4d615c0](https://github.com/alsmith151/ReguloNado/commit/4d615c0df5ca8495bd707aee87fe5e3f68aee70c))
* **attribution:** add ISM attribution and core-region calling ([4a5845a](https://github.com/alsmith151/ReguloNado/commit/4a5845aee4d58a2a51af7d5112a76ad0e97d5d05))
* **attribution:** add track-group targeting and post-discovery annotation ([7cfe547](https://github.com/alsmith151/ReguloNado/commit/7cfe547e4ed05554095c68f9b3b677197baa01d1))
* **borzoi:** enhance weight initialization for rotary frequencies in flash attention ([842484c](https://github.com/alsmith151/ReguloNado/commit/842484cdd920d7fca1b725ae799aa5a3fc3adfaa))
* **cli:** add `regulonado attribute` command ([88329c3](https://github.com/alsmith151/ReguloNado/commit/88329c3e2fc6ac869cc07f3e5a092f9fced9a60b))
* **config:** configure every model as a run with backbone, trunk, target and recipe ([80f8099](https://github.com/alsmith151/ReguloNado/commit/80f8099555484d2ff35ef523a698996702153020))
* **config:** enhance backbone architecture config handling for model weight loading ([300b423](https://github.com/alsmith151/ReguloNado/commit/300b423603c88c00fe68ebaea9e4d3a2d2331e21))
* **config:** enhance backbone architecture config handling for model weight loading ([ed2e8d3](https://github.com/alsmith151/ReguloNado/commit/ed2e8d3589f6e3c57b4591eed70047a546380aae))
* **config:** update GPU resource settings for train_phase and add design_shard and attribute_shard rules ([cbda53c](https://github.com/alsmith151/ReguloNado/commit/cbda53cd8a3f0bdb141831b442a2631039089f19))
* **counts:** port BAM region counting and RegionCountData with parquet storage ([01a68a2](https://github.com/alsmith151/ReguloNado/commit/01a68a20d665035b202e30c49757f633948540b7))
* **counts:** share one region table between counting and embedding ([4f827ee](https://github.com/alsmith151/ReguloNado/commit/4f827eead744bfb632c6c7871914db2bcd0f5323))
* **dataset:** write HF-layout Parquet shards, read by row group ([1fd2fb4](https://github.com/alsmith151/ReguloNado/commit/1fd2fb473fe524b1454b5e68fac681aead902daa))
* **design:** add bin reduction options and random seed support to design function; implement topk signal summaries and Hamming distance for design reporting ([9e89764](https://github.com/alsmith151/ReguloNado/commit/9e89764895670a1087bc02df7f15a15828751ff3))
* **design:** add checkpoint directories and holdout checkpoint to design configuration ([7489bfe](https://github.com/alsmith151/ReguloNado/commit/7489bfe0ca97d02bb1496b4557c9f9d3942e92b6))
* **design:** add exclude track option to design function and enhance track group resolution ([1239f53](https://github.com/alsmith151/ReguloNado/commit/1239f535d835c514de222062da04f569aed127d1))
* **design:** add from_attribution shorthand for candidates ([4babfc0](https://github.com/alsmith151/ReguloNado/commit/4babfc0b5e8e098adaee608b67ef56f8fbf8e977))
* **design:** add gain transform and pseudocount options for selective activation objective ([e453ece](https://github.com/alsmith151/ReguloNado/commit/e453ece371cb6d5903a9f0b8b250cf630fd62ae9))
* **design:** add score_pad_bp to widen the scored window independently of pad ([89bbacb](https://github.com/alsmith151/ReguloNado/commit/89bbacb9c6f0aade5b22c7c92c6f29734891ed60))
* **design:** add seed-relative selective activation objective with customizable tuning parameters ([ee33c76](https://github.com/alsmith151/ReguloNado/commit/ee33c76dbeda493f2ccd303704afddf3fc1963f2))
* **design:** add synthetic enhancer design module + CLAUDE.md ([93be35e](https://github.com/alsmith151/ReguloNado/commit/93be35e8428744c592b1646f95c115e6cfd681de))
* **design:** add threshold and rho options to AdaLead configuration ([6c1c781](https://github.com/alsmith151/ReguloNado/commit/6c1c78194b3dfe11fc45fa42f9fd6ae918fc52dc))
* **design:** add unique W&B run ID for design targets and configure default project naming ([8f72b79](https://github.com/alsmith151/ReguloNado/commit/8f72b7965adcbc32515b03c76180903d6f1fdea4))
* **design:** add validation for AdaLead budget constraints in DesignConfig ([f076baf](https://github.com/alsmith151/ReguloNado/commit/f076baff53bcfbcea27c5eac99d52e5fd3d001f1))
* **design:** add W&B group label option for design invocation and update project naming ([f266423](https://github.com/alsmith151/ReguloNado/commit/f266423a6f2eb5bcef225d2edf0e820d82553167))
* **design:** add Weights & Biases logging support to enhancer design and improve round metrics tracking ([35286b1](https://github.com/alsmith151/ReguloNado/commit/35286b1bc45eff6f3b5b05ec6b74551a8740027c))
* **design:** enhance logging and tracking in design function with wandb history management ([b5ffa50](https://github.com/alsmith151/ReguloNado/commit/b5ffa50405057a731a1bd886ed05e7dbbb829c7a))
* **design:** enhance trajectory logging and reporting in design module ([90af6d6](https://github.com/alsmith151/ReguloNado/commit/90af6d6acf0b1ce40af5285b93222720293698a6))
* **design:** implement selective activation objective with enhanced specificity metrics and reporting ([9629ec7](https://github.com/alsmith151/ReguloNado/commit/9629ec70569d7011c5aff204438eda5344b95f60))
* **design:** improve reproducibility and top-k diagnostics ([94ae890](https://github.com/alsmith151/ReguloNado/commit/94ae890a95c82f2689d90b06fc5af4e15d13e100))
* **design:** penalise drift from the seed sequence ([24105ed](https://github.com/alsmith151/ReguloNado/commit/24105ed39ddcd7b5a6e952812783bde70f97cf3c))
* **design:** refine settings merging for method-specific configurations ([76f2fa5](https://github.com/alsmith151/ReguloNado/commit/76f2fa5fc5a03cd7183e863f4b1c3dd3d1fdcbb7))
* **design:** support explicit crops and neutral flanks ([c3871ea](https://github.com/alsmith151/ReguloNado/commit/c3871ea58943292826ab5974e550d9070a26c8b1))
* **design:** support region-count ensembles downstream ([c15e489](https://github.com/alsmith151/ReguloNado/commit/c15e4896af5c9b95fc0afb6e46eb7b177cb27ef4))
* **embeddings:** add per-chromosome parquet embedding cache and embed regions CLI ([16886ea](https://github.com/alsmith151/ReguloNado/commit/16886ea0bd2bf8a316c313b7f493f22b3d1b77c3))
* **embeddings:** AlphaGenome encoder-only features for region-count runs ([800a74b](https://github.com/alsmith151/ReguloNado/commit/800a74bb483a4e5ac8fb554555b6a1cbbfff76b2))
* **embeddings:** check a cache fits in the job's memory before loading it into RAM ([c2c1875](https://github.com/alsmith151/ReguloNado/commit/c2c1875ecd9aa6d1854d0a4ae3333d0ef6202b4c))
* **embeddings:** store trunk embedding caches as Hugging Face datasets Arrow files ([2ac40dc](https://github.com/alsmith151/ReguloNado/commit/2ac40dc502375e89d8e9fc554fa3b3847cc10221))
* **examples:** add contrast-correlation deep-unfreeze pipeline config ([9a45a27](https://github.com/alsmith151/ReguloNado/commit/9a45a275b26a7a46e10858524e27a1b021e9daa9))
* expose anchor normalization in workflow ([600432c](https://github.com/alsmith151/ReguloNado/commit/600432c75d0fa491927b2a7ef4e266c4c3cd3150))
* **gather:** drop tracks without finite log_size_factor and warn user ([5f62d38](https://github.com/alsmith151/ReguloNado/commit/5f62d38d8e8114fd9a77c97f31eaae88942e9952))
* **inference:** add checkpoint validation to prevent loading unmerged adapter weights ([a1e3feb](https://github.com/alsmith151/ReguloNado/commit/a1e3febc50f8f271f6f669962a3ef7be882755e9))
* **losses:** add binwise Poisson-multinomial loss ([fc24ead](https://github.com/alsmith151/ReguloNado/commit/fc24eadafc8cee7919d69757f7dae95090146b84))
* **losses:** add contrast magnitude term to penalise shrunk cell-type differences ([3157614](https://github.com/alsmith151/ReguloNado/commit/315761461e86b2e87583cecbbc8420fc410c2aed))
* **metrics:** add design contrast objective to enhance metric evaluation ([5387d43](https://github.com/alsmith151/ReguloNado/commit/5387d4378a38551d9517dde9fa018b97c3a8805b))
* **model:** add adapter output geometry and AlphaGenome backbone adapter ([f748488](https://github.com/alsmith151/ReguloNado/commit/f7484882efa6fbf424480ac22f190dee5b0dd1e9))
* **model:** add pretrained_name to backbone architecture initialization ([d9289f0](https://github.com/alsmith151/ReguloNado/commit/d9289f05cd8338179ba87d7d46c782a5adf21ec6))
* **model:** add pretrained_name to backbone architecture initialization ([7fac5b0](https://github.com/alsmith151/ReguloNado/commit/7fac5b064d546d0501cf2cfc97f161adcac746cc))
* **model:** upgrade to HF PreTrainedModel and add BigWig prediction ([67c637c](https://github.com/alsmith151/ReguloNado/commit/67c637c8dec7324fde8c924c37846af811b86a79))
* **model:** upgrade to HF PreTrainedModel and add BigWig prediction ([69004a6](https://github.com/alsmith151/ReguloNado/commit/69004a61bcb6ce73f999fa7a3cce1ec54c9d4810))
* **parameter-sweep:** add dataset input to create_parameter_sweep rule ([bb4d0cb](https://github.com/alsmith151/ReguloNado/commit/bb4d0cbd3d0efdfc1d9a6100bff5396ade86a289))
* **pipeline:** add stage argument to pipeline function for selective execution ([9e03477](https://github.com/alsmith151/ReguloNado/commit/9e03477e01086f3fec96ee889e8c6be252bb4406))
* **predict:** add option to include one-hot input in RegionPrediction ([254ff3d](https://github.com/alsmith151/ReguloNado/commit/254ff3d925f0ae4249034970c316c5a64edc1eee))
* **predict:** add option to include one-hot input in RegionPrediction ([7815dfb](https://github.com/alsmith151/ReguloNado/commit/7815dfb8f8d6625427414f249679920695f779fe))
* **prediction:** add optional BigWig prediction stage and update documentation ([e3fd5a2](https://github.com/alsmith151/ReguloNado/commit/e3fd5a24c6182f610524b92755432b6ca63fd073))
* **predictor:** enhance batch processing with error handling and dynamic batch size adjustment ([4aaef24](https://github.com/alsmith151/ReguloNado/commit/4aaef2422055620d72863184c9bd11cacc359776))
* **regions:** add cached-embedding region dataset, HF Trainer runner and train-regions CLI ([d8bc4e4](https://github.com/alsmith151/ReguloNado/commit/d8bc4e47a156c53e1945a2adbe02e2ced37db92e))
* **regions:** add overfit_probe for training region diagnostics ([8d9e05c](https://github.com/alsmith151/ReguloNado/commit/8d9e05cd9b07ab62db13e3c5dfb0c49ff2c34937))
* **regions:** add region count head, NB likelihood loss and grouped count metrics ([31b54df](https://github.com/alsmith151/ReguloNado/commit/31b54df330e230f716328e1520a0d6cc30c32528))
* **regions:** data.overfit_probe trains and evaluates on the same regions ([ff1e41e](https://github.com/alsmith151/ReguloNado/commit/ff1e41e5f1e572810a228c08240a902984938f51))
* **regions:** fine-tune a live trunk on short inputs, and warm-start runs across runs ([9bc3278](https://github.com/alsmith151/ReguloNado/commit/9bc32781e6ab7e8f1a4674f530538ced0443c1e7))
* **regions:** load live count checkpoints for inference ([a6a7e67](https://github.com/alsmith151/ReguloNado/commit/a6a7e6752656ff0d091f0d79d34fe1ed34c44872))
* **regions:** per-cell-type pooling head (`model.pooling: per_group`) ([fd434a6](https://github.com/alsmith151/ReguloNado/commit/fd434a6d09a62ced3c763b8a4898bd4e177f8446))
* **regions:** region presets anneal on plateau with longer patience ([fcb1caa](https://github.com/alsmith151/ReguloNado/commit/fcb1caaec738b5f69cc8e664a176a2841db1e278))
* replace pyfastx with pyfaidx for genome sequence access ([5b8b1e7](https://github.com/alsmith151/ReguloNado/commit/5b8b1e743c66a5bba0c4f6003d0c5bdb979e7ec5))
* **runner:** add require_clip parameter to resolve_scale_and_clip for flexible clipping behavior ([a32ed43](https://github.com/alsmith151/ReguloNado/commit/a32ed433daf2fb07967c915925e2214324d01a62))
* **runner:** add seed parameter to training arguments for reproducibility ([243a5b4](https://github.com/alsmith151/ReguloNado/commit/243a5b44510876d54c784e6a0843948fcedb358a))
* **scripts:** add cross-track contrast replicate ceiling ([76eacaf](https://github.com/alsmith151/ReguloNado/commit/76eacaf0bf9617bcbb027a7cceecb2d3a33955ee))
* **scripts:** add installation script for flash-attn with support for prebuilt wheels ([e430e38](https://github.com/alsmith151/ReguloNado/commit/e430e3852e3f368de67049be172e19078deab650))
* **scripts:** add installation script for flash-attn with support for prebuilt wheels ([d629c07](https://github.com/alsmith151/ReguloNado/commit/d629c07634ccfb941ced9d447530fa4c214d2af7))
* **scripts:** update dataset tag handling in training script output directory ([525d670](https://github.com/alsmith151/ReguloNado/commit/525d670749cfbaee089562a0246583ee96f6fdd6))
* **scripts:** update dataset tag handling in training script output directory ([e4efc36](https://github.com/alsmith151/ReguloNado/commit/e4efc3610bf27b8a6b412af505fde5f0f7c379e0))
* **sharding:** add shard size options to control Arrow shard file count and size ([8598f85](https://github.com/alsmith151/ReguloNado/commit/8598f85e9300ff1772e12d34d7d599ef3a82e3fc))
* **sharding:** add shard size options to control Arrow shard file count and size ([8c622b2](https://github.com/alsmith151/ReguloNado/commit/8c622b220f579c3a612dadfd84f6ba07da612b68))
* **slurm:** add automatic job resubmission and dynamic resource scaling for retries ([3b7912f](https://github.com/alsmith151/ReguloNado/commit/3b7912f8f32c6545618e6194b9257ab473f48b66))
* **smoke-test:** add GPU smoke test scripts for flash_attn and Borzoi/FlashZoi ([a534490](https://github.com/alsmith151/ReguloNado/commit/a53449009e609d13d1267f956d0371ec63eea6a0))
* **sweep:** support overriding loss parameters in training settings ([bf45c64](https://github.com/alsmith151/ReguloNado/commit/bf45c64c02e26963023a05fd4b4d349953ebb63f))
* **tests:** add test for SequencePredictor to exclude auxiliary group channels ([6e5021e](https://github.com/alsmith151/ReguloNado/commit/6e5021eba94606ac366f1adb8a3bffb0fbf487e4))
* **tracks:** add fragment-lengths command to measure BAM fragment lengths ([a9f96b7](https://github.com/alsmith151/ReguloNado/commit/a9f96b78debea2a37281c376afab905a573b2d3e))
* **tracks:** add track discovery/QC/assemble pipeline ([d4969f3](https://github.com/alsmith151/ReguloNado/commit/d4969f3cfea44d06bc36f4fe3e6fba3a5ab135ae))
* **tracks:** discover tracks from BAMs as well as bigWigs ([3491ae1](https://github.com/alsmith151/ReguloNado/commit/3491ae14ca835cbdd67e620c99ad5cbd9a41e5f9))
* **tracks:** drop tracks the annotations don't list ([11e94ca](https://github.com/alsmith151/ReguloNado/commit/11e94ca3e56c79a30712a412a76a7d4f97abaffc))
* **tracks:** migrate build, normalization, and training to tracks.parquet ([fac4200](https://github.com/alsmith151/ReguloNado/commit/fac42001f7944addd776e0704f6d9018f98e9847))
* **train:** adjust GPU resource allocation for train_phase rule ([49aa8dd](https://github.com/alsmith151/ReguloNado/commit/49aa8dd2ff770ae2dd09eff600c12cdad2bd00de))
* **train:** enhance GPU visibility handling and subprocess environment for training ([cbfbf3b](https://github.com/alsmith151/ReguloNado/commit/cbfbf3b2086f94b2fcf2f601d1f8a5ae27546cc5))
* **trainer:** increase prefetch_factor to improve data loading efficiency ([ff21724](https://github.com/alsmith151/ReguloNado/commit/ff2172453b1c135a0d3e598ff8e31f42db700746))
* **training:** add a group-contrast head trained on anchor-unit specificity ([bb1e281](https://github.com/alsmith151/ReguloNado/commit/bb1e281b3cb25f449487c1ae1efd9379931002cb))
* **training:** add cross-track contrast loss and specificity metrics ([728378b](https://github.com/alsmith151/ReguloNado/commit/728378b0da8b77b7c10eac6076ccf2d4b292a926))
* **training:** add learnable per-track loss weighting ([9913467](https://github.com/alsmith151/ReguloNado/commit/991346742853b0d9ce2b14af703e5efb53a72efb))
* **training:** add LoRA/LoCon parameter-efficient trunk fine-tuning ([0224cde](https://github.com/alsmith151/ReguloNado/commit/0224cde253ee80902071606b58bcf8b95572e363))
* **training:** adjust num_workers for CPU allocation and add dataset schema validation ([a95423d](https://github.com/alsmith151/ReguloNado/commit/a95423dfbb8913034ae47e6c28f5c6280df476ef))
* **training:** enhance logging and add rank handling for distributed training ([15e9197](https://github.com/alsmith151/ReguloNado/commit/15e919735f70fcfcffc24e30075e21ae1e3cb3f5))
* **training:** enhance logging and add rank handling for distributed training ([02b4be5](https://github.com/alsmith151/ReguloNado/commit/02b4be5ae0e0b2ca0c8f3084214077b92cb4e769))
* **training:** rank contrast regions by strongest group; fix and trim eval metrics ([56a2e70](https://github.com/alsmith151/ReguloNado/commit/56a2e7056b7d7335db956d974c9bb46226f5746e))
* **training:** reduce-on-plateau learning-rate schedule ([6d1e6b9](https://github.com/alsmith151/ReguloNado/commit/6d1e6b9cbfd95bc8c52d7d9f566f9f2205a6872b))
* **training:** replace contrast KL with correlation loss on shared specificity stats ([18ac3d3](https://github.com/alsmith151/ReguloNado/commit/18ac3d36908934ff59184dcc522407695a56c2a4))
* **training:** score the group-contrast head against a specificity panel ([61121b8](https://github.com/alsmith151/ReguloNado/commit/61121b8ab23aff91018d28bc78d5de1a9135bb93))
* **training:** size streaming epochs from Arrow shard row counts ([0ed101b](https://github.com/alsmith151/ReguloNado/commit/0ed101b3569c6bdf040c667ac95bb3ef849dca4c))
* **training:** support group-conditioned FiLM sweeps ([134c803](https://github.com/alsmith151/ReguloNado/commit/134c803e447fc8ea09850de95e3b39cd647062d3))
* **training:** train on raw counts with a per-track exposure offset ([113069a](https://github.com/alsmith151/ReguloNado/commit/113069a93a1c0aaba63f1980578057e82c536279))
* update .gitignore to include additional dataset and example files ([4825dc5](https://github.com/alsmith151/ReguloNado/commit/4825dc5eb1cbf1179b25f7c7af33c52ee0b393ac))
* **workflow:** add optional attribution stage ([f51746a](https://github.com/alsmith151/ReguloNado/commit/f51746a66c78d08b314272cced95dc8e213d4c63))
* **workflow:** add Snakemake pipeline ([6d0ea40](https://github.com/alsmith151/ReguloNado/commit/6d0ea40f8a8ea13e37ca5f2ee15e482a2c940734))
* **workflow:** add Snakemake regions section for counts, embedding cache and region-head training ([99dbc9b](https://github.com/alsmith151/ReguloNado/commit/99dbc9bbd1706dd809601b93eb57915f06ed0882))
* **workflow:** cap every job's memory request at max_mem_mb (default 256 GB) ([4c23992](https://github.com/alsmith151/ReguloNado/commit/4c23992c264575e166e8fecc82f9ba14bd506523))
* **workflow:** implement candidate BED line handling for parquet tables ([1452b6b](https://github.com/alsmith151/ReguloNado/commit/1452b6b40af9dbbd6f08a3c6285ecd2aeb858fc5))
* **workflow:** named parameter sweeps, with live-trunk region trials ([32d219d](https://github.com/alsmith151/ReguloNado/commit/32d219d230ab9236311f2454de5e10641c791fe6))
* **workflow:** parameter sweeps over region-count heads ([03ae0a2](https://github.com/alsmith151/ReguloNado/commit/03ae0a244d702ab7b267a72570c95dedacb685dd))
* **workflow:** run downstream design from explicit checkpoints ([da6cb17](https://github.com/alsmith151/ReguloNado/commit/da6cb171ee6258e75e235cd6521cd993e37235b6))
* **workflow:** set every rule's memory in the rules, doubling on retry ([252e44d](https://github.com/alsmith151/ReguloNado/commit/252e44d246f709f8e177c6ce317bec698a2dcbd5))
* **workflow:** sweeps read small copies of their embedding caches ([bc9e9b7](https://github.com/alsmith151/ReguloNado/commit/bc9e9b70587e2a64e9eb4e924d585623b0d5196e))


### Bug Fixes

* **adapter:** handle varying transformer _initialize_weights signatures ([aba28ad](https://github.com/alsmith151/ReguloNado/commit/aba28adfe2192b3ae801bc3d8afe5f71c8d2e408))
* **attribute:** anchor cores on the true peak, not the segment centroid ([c168318](https://github.com/alsmith151/ReguloNado/commit/c168318b60b1f80ade5933f0f34e1c4048ef042d))
* **backbone:** fail on unset pretrained_name instead of silent random init ([d2b83ae](https://github.com/alsmith151/ReguloNado/commit/d2b83ae7cabfd2b48c4254cfe3b126ee2de9ff86))
* **backbone:** fail on unset pretrained_name instead of silent random init ([078893d](https://github.com/alsmith151/ReguloNado/commit/078893da9482d1758a9587955c8c7e10dce32397))
* **borzoi:** update _initialize_weights method to handle parameter name changes in transformers ([4717c2e](https://github.com/alsmith151/ReguloNado/commit/4717c2ea19ab53371ca646cf3e3cbb9013d1c5d5))
* **callbacks:** handle NaN/Inf values in axis limit calculations for plots ([e8972f4](https://github.com/alsmith151/ReguloNado/commit/e8972f4fbbc4715e7db04f768be3c63c1df450ab))
* **callbacks:** handle NaN/Inf values in axis limit calculations for plots ([ef49516](https://github.com/alsmith151/ReguloNado/commit/ef495165595bf890b4aa22e47571356705249a06))
* **config:** skip the anchor apply_squash check for count labels ([874193b](https://github.com/alsmith151/ReguloNado/commit/874193b2c35ce940734bcfc5c6e9f8aa468213d1))
* correct silent failure modes in metrics, prediction and FASTA reads ([1034399](https://github.com/alsmith151/ReguloNado/commit/1034399e003a4c746d457f67f587b605e34eb3e2))
* **dataset:** correct splits path formatting in build_dataset rule ([68b85b8](https://github.com/alsmith151/ReguloNado/commit/68b85b8711d50f5ad26128bcfee13813b57674ea))
* **design:** rebuild attribution BigWigs from explicit positions ([0166c48](https://github.com/alsmith151/ReguloNado/commit/0166c486495fa3e8a97e2de283d9dbd561f8854b))
* **embeddings:** load in-memory caches without datasets hashing their contents ([3070558](https://github.com/alsmith151/ReguloNado/commit/307055824070a3e49feaa1e1370ed2430488f7bb))
* **embeddings:** use per-process temp name when writing the manifest ([f1bcd55](https://github.com/alsmith151/ReguloNado/commit/f1bcd558273a4e16d857dda9744b5a9fd6fed15e))
* **model:** keep frozen backbone BatchNorm in eval mode during training ([2132eab](https://github.com/alsmith151/ReguloNado/commit/2132eab3182504336c1c84eb760d49948e4ec9c1))
* **model:** match head dtype instead of hardcoding float32 ([cbdfd8d](https://github.com/alsmith151/ReguloNado/commit/cbdfd8dfcb2bc333c3a627d1cbd458d2ac81ab23))
* **model:** restore Borzoi relative-position buffer on load ([e797166](https://github.com/alsmith151/ReguloNado/commit/e79716633897a85f8c8f7644d7b46f2d1a761a3d))
* **model:** stop post_init erasing head initialisation; seed output bias for all heads ([60f5de0](https://github.com/alsmith151/ReguloNado/commit/60f5de0412d3d8b6e60fd6dfcc75f66445605eb0))
* **model:** stop share_condition_base_channels being a no-op ([9913467](https://github.com/alsmith151/ReguloNado/commit/991346742853b0d9ce2b14af703e5efb53a72efb))
* **model:** train flashzoi with fp32 weights under bf16 autocast ([d388673](https://github.com/alsmith151/ReguloNado/commit/d388673dde34f8d4b860457a5152cbc4809fb1ea))
* **normalization:** correct TMM to match edgeR reference implementation ([3ebd2df](https://github.com/alsmith151/ReguloNado/commit/3ebd2dfc71e183c0f562d9c62848c881ec12c11c))
* **recompress:** raise instead of calling sys.exit from library code ([a78266e](https://github.com/alsmith151/ReguloNado/commit/a78266e5324ae409589a0bb469aecd675b7a425a))
* **regions:** DataLoader workers share batches through the file system ([c911f63](https://github.com/alsmith151/ReguloNado/commit/c911f631309ce6a528c0261e0aebfde2155527c8))
* remove remaining cluster-specific paths from shipped code ([9ee67b4](https://github.com/alsmith151/ReguloNado/commit/9ee67b4b2757b64b21821fe2c6264654a328ec58))
* **runner:** enable ddp_find_unused_parameters to handle trainable parameters in metadata heads ([713158f](https://github.com/alsmith151/ReguloNado/commit/713158f38dd8c32f80842f6bb4bd7606e3b89eb0))
* **search:** cast energy to float32 before converting to numpy array to handle bf16 ([2277c1c](https://github.com/alsmith151/ReguloNado/commit/2277c1cfd1c5dbaa745b14c8595ae67301905f88))
* **slurm:** require explicit REPO_DIR instead of unreliable BASH_SOURCE ([26dad37](https://github.com/alsmith151/ReguloNado/commit/26dad37d837d84d90c578e718cfbb9a009e32fed))
* **slurm:** require explicit REPO_DIR instead of unreliable BASH_SOURCE ([1283ec1](https://github.com/alsmith151/ReguloNado/commit/1283ec1fa884bf188ce0a601e0e7ec3691b08f11))
* **tracks:** keep measuring fragment lengths past a BAM without usable reads ([0d0edd3](https://github.com/alsmith151/ReguloNado/commit/0d0edd3502067bab04f64294ca66ea9c3195c23d))
* **tracks:** split clip thresholds into unit-specific column families ([f2b86e8](https://github.com/alsmith151/ReguloNado/commit/f2b86e8d4104d504a6e8decb58e62f1e53322324))
* **train:** avoid torchrun port collisions; make recompress-dataset resumable ([eb059c6](https://github.com/alsmith151/ReguloNado/commit/eb059c6c43fc551bed10039565502cc5a0b1dd34))
* **train:** declare matplotlib and degrade gracefully without it ([a0f5bd3](https://github.com/alsmith151/ReguloNado/commit/a0f5bd3d91d91aef22b122b17ce577cc9873f4af))
* **training:** accumulate eval sufficient statistics in float32 ([987a81c](https://github.com/alsmith151/ReguloNado/commit/987a81c8a16f91943775fcf34dc182232f64cd7e))
* **training:** allow warm-starts across learn_track_weights settings ([27d2af9](https://github.com/alsmith151/ReguloNado/commit/27d2af981a80f6da478c172f52e74c9edfdb77a9))
* **training:** bound sweep data-loader memory ([17acd38](https://github.com/alsmith151/ReguloNado/commit/17acd388946d3c7adea6fde375b7362a6c50e928))
* **training:** ensure empirical head initialization occurs before dataset transformations ([3fdda98](https://github.com/alsmith151/ReguloNado/commit/3fdda98a95c65142b39c68a37258fd38d03dfac6))
* **training:** label logged learning rates by parameter family ([5a03ac5](https://github.com/alsmith151/ReguloNado/commit/5a03ac5345f15894c64dddea550c4253062df6c6))
* **training:** validate checkpoint/eval step alignment and rebase preset stages ([8de2a69](https://github.com/alsmith151/ReguloNado/commit/8de2a69700ebdf8a13c7ef4bbb3a6ed3359a0436))
* update Python version constraint and clean up GPU dependencies ([a386324](https://github.com/alsmith151/ReguloNado/commit/a386324d240be746ae5ad015b42a6c6395aba16e))
* **workflow:** invoke renamed dataset subcommand in build_dataset rule ([4fdb05d](https://github.com/alsmith151/ReguloNado/commit/4fdb05d24cd24f52a14556e1e9da5009573102e0))
* **workflow:** keep final dataset path stable ([5fef83c](https://github.com/alsmith151/ReguloNado/commit/5fef83cb7ea8e2dbe39dba7edff94ee8635f80ef))
* **workflow:** never request less than 8 GB for a job ([da8548f](https://github.com/alsmith151/ReguloNado/commit/da8548f6db56611e22ffa0bc34f112ea41c4d24f))
* **workflow:** omit unused design topk setting ([724b085](https://github.com/alsmith151/ReguloNado/commit/724b085199d3d9ee263e8cf370701df3006a74ed))
* **workflow:** resolve design shard count via checkpoint ([84b7bf0](https://github.com/alsmith151/ReguloNado/commit/84b7bf074d5465960d27834653cd6c0a2dd5243f))


### Performance Improvements

* **embeddings:** stitch straddling regions across tiles and stream tiles/row groups ([a6806ab](https://github.com/alsmith151/ReguloNado/commit/a6806ab65cc49f7208449f40efec233b51b4a882))
* **tracks:** read BAMs in parallel processes and report progress in fragment-lengths ([e711b26](https://github.com/alsmith151/ReguloNado/commit/e711b26eaf0ffd26b9eb50b53488503278612c58))


### Documentation

* add BamNado fold training config ([3804ff9](https://github.com/alsmith151/ReguloNado/commit/3804ff9d4ffca2b80af6a67f2a53a821b82b5504))
* **attribution:** document core pre-selection and the design chain ([ac69fc3](https://github.com/alsmith151/ReguloNado/commit/ac69fc358c0f97739849cb20f3a77b483ef79758))
* describe runs, recipes, trunks and targets ([fb91507](https://github.com/alsmith151/ReguloNado/commit/fb91507653b5af31dc2af7191026cf8503d89a2c))
* document anchor normalization ([1d5af69](https://github.com/alsmith151/ReguloNado/commit/1d5af691772d01918d0eaf568c362736e3a7b93d))
* document debug-writers feature build command ([f553c04](https://github.com/alsmith151/ReguloNado/commit/f553c04552696e4c9615f032e4177afc675c0050))
* document debug-writers feature build command ([a5edb20](https://github.com/alsmith151/ReguloNado/commit/a5edb20def11d84a81ca45c3b9d243ff97ec0e5d))
* document region-count modelling on cached embeddings ([6686e87](https://github.com/alsmith151/ReguloNado/commit/6686e87425dffa03fb70531a18735716f46d135c))
* document stable public API, expand checkpoint/prediction docs ([d643353](https://github.com/alsmith151/ReguloNado/commit/d6433534d45f9f41cd3c5581db44f7143d541adf))
* document the Python API and add ARCHITECTURE/CONTRIBUTING ([b7ee643](https://github.com/alsmith151/ReguloNado/commit/b7ee643325b8b4d95c3cca120f240008c33ad8ee))
* document the track table system and QC, add HL-60 examples ([420268b](https://github.com/alsmith151/ReguloNado/commit/420268bb2b68d518cb0a322c9af4aa2fab5e4515))
* explain region-count design and attribution ([ec115a9](https://github.com/alsmith151/ReguloNado/commit/ec115a95266b6bb9aefe6b0986631dbe1cccbe4e))
* explain the Snakemake pipeline ([b90cadc](https://github.com/alsmith151/ReguloNado/commit/b90cadcc4101a3c541031e45853f70f4a51722d0))
* record successful alpha consolidation checks ([8ebe4bb](https://github.com/alsmith151/ReguloNado/commit/8ebe4bbfb6f9518bb12c3beb153ac01818cb42a5))
* rewrite README for the Snakemake workflow ([a0f7289](https://github.com/alsmith151/ReguloNado/commit/a0f72898b3b97bf13bd75062c623c3a3604d7f73))
* **rust:** document the production writer and remaining public items ([74a202f](https://github.com/alsmith151/ReguloNado/commit/74a202f17d909c979269fd53463c32eb72ab32d3))
* update README to include BAM file support and clarify model training details ([16ee94b](https://github.com/alsmith151/ReguloNado/commit/16ee94bde2cf5858f47a747df13ddda48fa60e1e))


### Code Refactoring

* **training:** rename the region-count trainer from "cached" to "regions" ([0a28110](https://github.com/alsmith151/ReguloNado/commit/0a28110f605eab5f4afc48a2c5617007caa549a8))
* **training:** train cached-trunk heads through `regulonado train` ([21d0b18](https://github.com/alsmith151/ReguloNado/commit/21d0b185743dce9c7afbdae2345f0aa43cdb6ab3))

## [Unreleased]

### Changed

- **Workflow config (breaking):** every model is a run under `train.runs`, choosing its
  `backbone` (`{type, pretrained}`), `trunk` (`live` | `cached`), `target` (`profile` |
  `region_counts`) and `recipe`, a named phase chain under `train.recipes`. What a run
  predicts is configured under `targets`, and each dataset is built only if a run uses
  it:
  - `train.phases` → `train.recipes.<name>`; `runs[].pretrained_model` →
    `runs[].recipe` + `runs[].backbone`;
  - `inputs.intervals` + `dataset:` → `targets.profile`;
    `dataset.drop_missing`/`dedupe_tracks` → `inputs`;
  - tracks are discovered from bigWigs when `targets.profile` is set, otherwise from BAMs,
    into one `results/tracks/tracks.parquet`;
  - prediction, attribution and design only accept `target: profile` runs, and default to
    all of them;
  - examples are migrated. `hl60_anchor_folds`, `enhancer_candidates_anchor_folds` and
    `hl60_design` also drop their obsolete dotted `data.apply_squash` key.
- Renamed the `regulonado build` CLI command to `regulonado dataset` — the
  name now matches what it produces.
- Split monolithic dependency list into focused extras: `data`, `train`, `gpu`,
  `viz`, `dev`, `jupyter`, `test`, `all`.
- Widened Python version range to `>=3.12,<3.14` for compatibility with the
  wider nadoverse ecosystem (nadoverse, quantnado).
- Bumped `numpy` to `>=2` (aligns with bamnado + quantnado).
- Bumped `pysam` to `>=0.23` (aligns with quantnado).
- Removed `torch<2.7` upper bound (allows co-installation with packages that
  pin later torch versions).
- Removed `lightning<3` upper bound.
- Removed `flash_attn<2.8` upper bound.
- Moved `ruff` and `ninja` from core deps to `[dev]`/`[gpu]` extras.
- Made top-level model imports in `__init__.py` lazy — core install no longer
  requires torch, borzoi-pytorch, or enformer-pytorch.
- Added `__version__` to the top-level namespace.
- Added `[project.urls]` with repository link.
- `EnformerBackboneAdapter.feature_dim` now reports `2 * dim` (3072), which matches what
  `return_only_embeddings=True` actually returns (it was previously `dim`).

### Added

- Region-count modelling on cached trunk embeddings (`docs/region-counts.md`): a
  `trunk: cached`, `target: region_counts` run trains an NB count head on embeddings
  its pretrained trunk computed once:
  - `regulonado counts regions/bam/gather`: BAM region counting, ported from UEF, with
    anchor-based size factors, stored as parquet. `counts regions` writes the canonical
    region table that counting and embedding share, so embedding caches build while BAMs
    are still being counted. A region set's own `target_start`/`target_end` are kept;
  - `regulonado embed regions`: a backbone-agnostic per-chromosome parquet embedding
    cache. In the workflow, one cache per distinct trunk setup, shared by every run
    that uses it;
  - `regulonado train --trunk cached --embeddings DIR`, with `pretrain`/`specific`/`target`
    presets (`python/configs/cached_experiment/`, over `train_cached.yaml`);
  - `data.exclude_regions` / `targets.region_counts.exclude_regions`: held-out
    candidates leave training at every phase (UEF `--exclude_bed`).
- `regulonado tracks discover --format bam`: tracks discovered from BAMs share the
  bigWig tracks' `discover`/`assemble` steps and `tracks.parquet` schema. `--bam-dir`
  finds BAMs by `sample_id` (a whole-token match, so run accessions work) and, for bigWig
  discovery, records each track's `bam`. Track sheets and SeqNado projects now carry
  `bam` into `tracks.parquet`. bigWig-only commands refuse a BAM track table.
- `regulonado pipeline` composes every run × recipe phase before scheduling, and the
  cached-trunk config rejects keys `train_cached.yaml` does not declare.
- `AlphaGenomeBackboneAdapter` (`alphagenome` extra), and adapter output geometry
  (`output_bin_size`, `fixed_input_length`, `input_multiple`, `output_span`).
- `regulonado.sequence`: pid-safe pyfastx genome access and zero-padded window fetches.
- GitHub Actions CI workflow (`ci.yml`): Python 3.12/3.13 matrix, Rust
  toolchain, pytest over data/chrom-pass tests (no GPU required).
- GitHub Actions release workflow (`release.yml`): maturin wheel matrix
  (Linux x86_64/aarch64, macOS x86_64/aarch64) + PyPI OIDC Trusted Publisher.
- Automated versioning via release-please (`release-please.yml`,
  `release-please-config.json`, `.release-please-manifest.json`).

---

## [0.1.0] — 2026-06-22

Initial versioned release.

### Added

- Rust/PyO3 chromosome-pass Arrow writer (`chrom_pass`).
- `regulonado build` CLI for Arrow DatasetDict construction from BED/FASTA/BigWig.
- `regulonado train` CLI delegating to Hydra + HF Trainer.
- `regulonado scale` and dataset utility subcommands.
- Borzoi and Enformer backbone adapters.
- FiLM, ResidualFiLM, and TransferMLP prediction heads.
- Poisson-multinomial and related loss functions.
- Per-run provenance outputs (`provenance.json`, `resolved_config.json`,
  `git_diff.patch`).
