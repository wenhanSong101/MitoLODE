# Virtual Perturbation Module

This folder provides the **virtual perturbation** experiments built on top of
a trained MitoLODE model. Two complementary directions are supported:

| Script                        | Direction                     | Goal |
|-------------------------------|-------------------------------|------|
| `perturbation_forward.py`     | Disease &rarr; Control         | find which genes, when corrected at which time window, shift the LRRK2 trajectory toward the healthy Control Day42 state |
| `perturbation_reverse.py`     | Control &rarr; Disease         | find which genes, when perturbed in the healthy state, drive the trajectory toward disease (complementary, sanity-check direction) |

Both scripts share a four-layer analysis design:

- **Layer 1A** &mdash; single-gene scan (984 mitochondrial genes)
- **Layer 1B** &mdash; pathway-level cascade perturbation (10 curated pathway groups)
- **Layer 1C** &mdash; two-pathway combined perturbation, with synergy score
- **Layer 2**  &mdash; gradient-guided optimal perturbation (3000 Adam steps, L2 + sparsity + KG prior regularizers)

## Prerequisites

1. A trained MitoLODE checkpoint `best_by_r_lrrk2.pt` produced by
   `mitolode/train.py --dataset lrrk2`.
2. The released MitoKG `gnn/` directory containing
   `gene_embeddings.npy`, `gene_list.txt`.
3. The processed LRRK2 dataset `data/lrrk2/` with `expr_scaled.npy` and
   `cell_meta.csv`.

The `mitolode` package must be importable; the easiest way is to run the
script from the project root, or add `mitolode/` to `PYTHONPATH`.

## Usage

Forward perturbation (Disease &rarr; Control):

```bash
python perturbation/perturbation_forward.py \
    --data-dir ./data/lrrk2 \
    --gnn-dir  ./MitoKG/gnn \
    --weight   ./output_lrrk2/best_by_r_lrrk2.pt \
    --out-dir  ./output_perturbation_forward
```

Reverse perturbation (Control &rarr; Disease):

```bash
python perturbation/perturbation_reverse.py \
    --data-dir ./data/lrrk2 \
    --gnn-dir  ./MitoKG/gnn \
    --weight   ./output_lrrk2/best_by_r_lrrk2.pt \
    --out-dir  ./output_perturbation_reverse
```

Default `--seed` is `99`, matching the paper.

## Outputs

### Forward

- `fig01_baseline_summary.png`
- `fig02_{Day0,Day10,Day14}_layer1a_topgenes.png`
- `fig03_{Day0,Day10,Day14}_layer1b_pathway.png`
- `fig04_timewindow_gene_heatmap.png`
- `fig05_gradient_convergence.png`
- `fig06_{Day0,Day10,Day14}_layer2_targets.png`
- `fig07_timewindow_comparison.png`
- `fig08_{Day0,Day10,Day14}_layer2_heatmap.png`
- **`perturbation_lrrk2_temporal.xlsx`** &mdash; 7-sheet workbook; **consumed
  directly by the MGMI inference module** via its `L2_GradientTargets` sheet.

### Reverse

- `lrrk2_fig1_layer1a_{Day0,Day10,Day14}` &mdash; Replace and Knockout side-by-side
- `lrrk2_fig2_layer1b_{Day0,Day10,Day14}` &mdash; pathway cascade
- `lrrk2_fig3_timewindow` &mdash; gene x timepoint heatmap
- `lrrk2_fig4_known_pd_genes_summary` &mdash; known PD genes across all windows
- `reverse_perturbation_results.xlsx`

All figures are saved in PNG, SVG, and PDF (reverse) / PNG only (forward).

## Key hyperparameters (hard-coded to the paper setting)

```
TOP_SCAN_GENES   = 984     # Layer 1A scans all mitochondrial genes
L2_STEPS         = 3000    # Layer 2 gradient optimization
L2_LR            = 0.005
L2_LAMBDA_SPARSE = 0.01    # L1 sparsity on delta
L2_LAMBDA_KG     = 0.10    # KG-prior: penalize edits on non-MitoKG genes
L2_TOP_N         = 20      # top-N targets reported
```

Known PD causal genes (PINK1, LRRK2, PRKN, PARK7, SNCA, HTRA2, TOMM20,
TOMM40) are **always** reported in the reverse experiment regardless of
rank.

## Note on the KG prior

The `L2_LAMBDA_KG` term encourages edits to concentrate on genes that are
present in the MitoKG pathway groups. Removing it makes the optimization
drift toward generic, biologically less interpretable edits; this is a
deliberate design choice and matches the paper.

## Pipeline position

```
train.py (LRRK2)
    |
    v
perturbation_forward.py  ->  perturbation_lrrk2_temporal.xlsx
    |
    v
mgmi_inference.py        ->  mitokg_llm_raw_v2.json
    |
    v
mgmi_external_validation.py  -> mgmi_external_validation.csv
```
