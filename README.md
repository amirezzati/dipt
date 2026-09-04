# DIPT — Domain-Invariant Prompt Tuning for Knowledge Distillation in Computational Pathology

Reference implementation of *"All Centers Are at most a Few Tokens Apart: Knowledge
Distillation with Domain Invariant Prompt Tuning"* (Ezzati et al.), plus the
follow-up experiments: **QuiltNet** as a second vision–language teacher and
**self-distillation**, where the student is the teacher's own image encoder.

Domain generalisation in computational pathology suffers from staining, scanner
and protocol shifts between clinical centers. Unlike natural images, histopathology
centers have no semantic descriptor ("a sketch of…"), so domain-specific prompts
cannot be written by hand. DIPT learns them from data instead:

1. **Stage 1 — DIPT.** Learn `K` continuous context tokens *per center*, anchored to
   a class-generic embedding built from hand-written templates.
2. **Aggregation.** Encode each center's prompt with the VLM text encoder and average
   across centers → a *domain-invariant, class-generic* text embedding.
3. **Stage 2 — Distillation.** Feed that embedding to **RISE** or **VL2V-ADiP** as the
   text target the student's visual features are aligned to.

> **[docs/APPROACH.md](docs/APPROACH.md)** is the full guide: the method in
> detail, what each module does, how the stages hand artifacts to each other, the
> exact order to run things, and a complete flag reference. Start there if you are
> new to the repository.

---

## Repository layout

```
dipt-cpath/
├── docs/
│   └── APPROACH.md               full walkthrough: method, structure, run order
├── configs/
│   └── class_templates.json      hand-written prompt templates per class
├── dipt/                         the library
│   ├── cli.py                    shared argparse blocks (--vlm, --student, …)
│   ├── paths.py                  repo-relative path resolution
│   ├── data/                     dataset registry + MultipleDomainDataset
│   ├── models/                   VLM registry, prompt learner, student backbones
│   ├── prompts/                  template aggregation, learned-prompt aggregation
│   ├── methods/                  dipt.py · rise.py · vl2v_adip.py
│   ├── evaluation/               metrics + evaluation loops
│   └── utils/                    logging, seeding, small helpers
├── scripts/                      entry points (one job each)
│   ├── download_datasets.py      download + prepare data
│   ├── prepare_kather_domains.py Kather19 pseudo-domains via stain augmentation
│   ├── check_setup.py            verify install and data layout
│   ├── train_domain_prompts.py   stage 1 — DIPT
│   ├── train_rise.py             stage 2 — RISE
│   ├── train_vl2v_adip.py        stage 2 — VL2V-ADiP
│   ├── evaluate_students.py      test RISE / self-distilled students
│   ├── evaluate_vl2v_adip.py     test VL2V-ADiP models
│   └── evaluate_zero_shot.py     zero-shot baselines
├── experiments/                  ready-to-run shell wrappers (00 … 06)
├── data/                         datasets (git-ignored, created by the scripts)
└── outputs/                      checkpoints, logs, results (git-ignored)
```

Every path defaults to a location **relative to the repository root**, so the code
runs unchanged on any machine. Override with `--data-root` / `--output-root`, or
with the `DIPT_DATA_ROOT` / `DIPT_OUTPUT_ROOT` environment variables.

---

## Install

```bash
conda create -n dipt python=3.10 -y && conda activate dipt
pip install -r requirements.txt
python scripts/check_setup.py
```

`scripts/check_setup.py` reports missing packages, available GPUs, the registered
teachers/students, and which dataset domains are present on disk.

---

## Datasets

### Camelyon17-WILDS

```bash
python scripts/download_datasets.py --dataset camelyon17
```

This downloads the WILDS release and **re-groups the patches by hospital using the
`center` column of `metadata.csv`, deliberately ignoring the official
train/val/test split** — domain generalisation here means holding out an entire
center. The result:

```
data/camelyon17/<center 0-4>/<normal|tumor>/patch_patient_XXX_node_Y_x_A_y_B.png
```

Patches are hardlinked from the raw download (instant, no extra disk). Use
`--link-mode copy` on a filesystem that disallows hardlinks, and
`--max-per-class-per-center N` to build a smaller subset for a quick trial.

Center **1** is reserved for validation in both stages (see `dipt/data/registry.py`),
and the four leave-one-center-out splits are `{0,2,3}→4`, `{0,2,4}→3`,
`{0,3,4}→2`, `{2,3,4}→0`.

### Kather19

```bash
python scripts/download_datasets.py --dataset kather19
python scripts/prepare_kather_domains.py --num-domains 6      # needs: pip install staintools spams
```

Kather19 ships no per-center metadata, so — as in the thesis experiments — the
pseudo-centers are synthesised by stain augmentation: each patch is re-stained
with one fixed `(alpha, beta)` perturbation per pseudo-domain, giving

```
data/kather19/<domain 0-5>/<ADI|BACK|DEB|LYM|MUC|MUS|NORM|STR|TUM>/*.tif
```

---

## Running the pipeline

Two teachers are supported everywhere via `--vlm`:

| `--vlm`    | HuggingFace checkpoint      |
|------------|-----------------------------|
| `plip`     | `vinid/plip`                |
| `quiltnet` | `wisdomik/QuiltNet-B-32`    |

### Stage 0 — zero-shot inference

Sweeps every teacher and prompt source in one run and prints a comparison table:

```bash
python scripts/evaluate_zero_shot.py                       # plip + quiltnet, template + handcrafted
python scripts/evaluate_zero_shot.py --vlm plip --prompt-source template
python scripts/evaluate_zero_shot.py --vlm all --prompt-source dipt -k 4
```

```
teacher    prompts                 d0            d1            d2            d3            d4          mean         worst
plip       template       ...
quiltnet   template       ...
```

### Stage 1 — learn the domain prompts

```bash
python scripts/train_domain_prompts.py --vlm plip --domain all --num-context-tokens 4
```

`--domain all` loops over every non-validation center. Checkpoints go to a
deterministic location, which is how later stages find them without any path
configuration:

```
outputs/prompts/<dataset>/<vlm>/k<K>/domain<D>/best_prompt_learner.pth
```

Check the learned prompts before spending GPU time on distillation:

```bash
python scripts/evaluate_zero_shot.py --vlm plip --prompt-source dipt -k 4
```

### Stage 2 — distillation

The prompt source is a flag, so the paper's baseline and the DIPT variant are the
same script:

| `--prompt-source` | class text embeddings                              | corresponds to    |
|-------------------|----------------------------------------------------|-------------------|
| `dipt`            | aggregated learned per-center prompts              | **ours**          |
| `template`        | aggregated hand-written templates                  | original RISE     |
| `handcrafted`     | one prompt per class                               | original VL2V-ADiP|

```bash
# RISE + DIPT, ResNet-50 student, all four splits
python scripts/train_rise.py --vlm plip --prompt-source dipt -k 4

# VL2V-ADiP + DIPT, ViT-B/16 student (runs its projection stage then its encoder stage)
python scripts/train_vl2v_adip.py --vlm plip --student vit_base --prompt-source dipt -k 4

# original baselines
python scripts/train_rise.py --vlm plip --prompt-source template
python scripts/train_vl2v_adip.py --vlm plip --prompt-source handcrafted
```

### Self-distillation

Setting `--student vlm` makes the student a **trainable copy of the teacher's own
image encoder**, fine-tuned so its visual features align with the domain-invariant
text embeddings. Same scripts, one flag:

```bash
python scripts/train_rise.py       --vlm quiltnet --student vlm --prompt-source dipt -k 3
python scripts/train_vl2v_adip.py  --vlm quiltnet --student vlm --prompt-source dipt -k 3
```

> Earlier experiments reported numbers for a *separate* student model. In the
> self-distillation setting the reported model **is** the fine-tuned image encoder
> of the VLM, evaluated on the held-out center.

### Evaluation

```bash
python scripts/evaluate_students.py \
    --run-dir outputs/rise/camelyon17/plip/resnet50_bit/dipt_k4 \
    --vlm plip --student resnet50_bit

python scripts/evaluate_vl2v_adip.py \
    --run-dir outputs/vl2v_adip/camelyon17/plip/resnet50/dipt_k4
```

The test center is inferred from each checkpoint name (`t_0_2_3_v_1_best.pth` →
test on center 4); mean and worst-case accuracy/F1 across splits are printed and
written to `evaluation.json` / `evaluation.txt`.

### Shell wrappers

`experiments/` holds thin wrappers with the paper's defaults. Everything is an
environment variable:

```bash
VLM=plip     K=4 bash experiments/01_domain_prompts.sh
VLM=plip     K=4 bash experiments/02_rise.sh
VLM=plip     K=4 bash experiments/03_vl2v_adip.sh
VLM=quiltnet K=3 bash experiments/04_self_distill_rise.sh
VLM=quiltnet K=3 bash experiments/05_self_distill_vl2v_adip.sh
VLM=plip     K=4 STUDENT=resnet50_bit bash experiments/06_evaluate.sh
```

A full PLIP pipeline end to end:

```bash
python scripts/download_datasets.py --dataset camelyon17
VLM=plip K=4 bash experiments/01_domain_prompts.sh
VLM=plip K=4 bash experiments/02_rise.sh
VLM=plip K=4 bash experiments/06_evaluate.sh
```

---

## Methods at a glance

**Stage 1 — DIPT** (`dipt/methods/dipt.py`). Prompt for class *i* of one center:
`[SOT] c_1 … c_K a_i [EOT]`, where `c_*` are learnable and `a_i` is the frozen
aggregated template embedding. Only the context tokens are trained:

```
L = CE(logits, y) + score_weight · (1 − cos(learned text feature, a))
```

The drift term is what keeps per-center prompts close enough to be averaged.

**Stage 2a — RISE** (`dipt/methods/rise.py`):

```
L = w_kd · KL(student/T ‖ teacher/T) · T²
  + w_cls · CE(student logits, y)
  + w_dist · (1 − cos(student feature, E_y))
```

**Stage 2b — VL2V-ADiP** (`dipt/methods/vl2v_adip.py`), two stages with the dual
feature-consistency loss

```
L = −λ · cos(proj, teacher image feature) − (1 − λ) · cos(proj, E_y)
```

stage 1 trains the projection layer only; stage 2 unfreezes the student encoder.
The classifier is a frozen zero-shot head built from `E`.

Default hyper-parameters follow the paper and the original runs: `K ∈ {3,4,5,6}`,
prompt lr `5e-5` with drift weight `0.5`; RISE lr `8e-4` with
`(w_kd, w_cls, w_dist) = (0.3, 0.4, 0.3)` and `T = 2`; VL2V-ADiP `2500` steps per
stage at lr `5e-5` with `λ = 0.5`. Self-distillation runs drop the KL term
(`0.0, 0.6, 0.4`), since teacher and student share an architecture.

---

## Notes on this reimplementation

The code was consolidated from three separate research trees. Behaviour is
preserved; the following are worth knowing:

* **Prompt templates.** `configs/class_templates.json` reproduces the original
  `utils/avg_template_prompts.py` verbatim (22 `normal` + 26 `tumor` templates),
  including a source typo that concatenated three prompts into one string — kept
  so the aggregated embedding matches the published runs. Aggregation averages raw
  embeddings and normalises once, as the original did. Kather19 templates are new.
  See [docs/APPROACH.md §8](docs/APPROACH.md#8-deviations-from-the-original-code).
* **Student head initialisation.** The original RISE scripts contained
  `student.fc.weight.datadata = mean_prompts`, a typo that silently did nothing, so
  the head was in fact randomly initialised. That behaviour is the default here;
  `--init-head-from-prompts` enables the intended initialisation.
* **Preprocessing.** Teacher and student resize to 224 and normalise on-device
  (bicubic + CLIP/ImageNet statistics) instead of round-tripping every batch through
  PIL and `CLIPProcessor`. Same statistics, substantially faster.
* **Vendored `timm`.** The old tree shipped a full copy of `timm`; this repo depends
  on the released package instead.

## Citation

```bibtex
@inproceedings{ezzati2025dipt,
  title     = {All Centers Are at most a Few Tokens Apart: Knowledge Distillation
               with Domain Invariant Prompt Tuning},
  author    = {Ezzati, Amir Mohammad and Malekhosseini, Alireza and
               Khosravi, Armin and Rohban, Mohammad Hossein},
  year      = {2025}
}
```
