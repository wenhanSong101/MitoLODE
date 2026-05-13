# MitoLODE Main Experiment

Training and evaluation code for the MitoLODE main experiment on the
PINK1 and LRRK2 datasets.

This release contains the **main-experiment configuration only**. Ablation
studies (component removal, mask-ratio sweep, train-ratio sweep, etc.)
are not included in this public release. If you need the per-dataset
tuned hyperparameters or the ablation scripts, please contact the authors.

## Files

| File | Purpose |
|---|---|
| `model.py` | MitoLODE model (KGSFM, KG-ODE, VAE encoder/decoder, losses) |
| `train.py` | Main-experiment training + evaluation + 6 diagnostic figures |

## Installation

```bash
pip install torch torchdiffeq numpy pandas scikit-learn scipy matplotlib seaborn
```

## Required data

You need two data directories:

### `<data-dir>` — dataset to train on
Must contain:
- `expr_scaled.npy` — (N_cells, N_genes) float32 z-scored expression matrix
- `cell_meta.csv`   — per-cell metadata

Expected metadata format:
- **PINK1**: a `condition` (or `stim`) column with values like `PINK1_D06` or `Control_IPSCs`; the script splits this into `condition` (`PINK1`/`Control`) and `timepoint` (`IPSCs`/`D06`/`D15`/`D21`).
- **LRRK2**: standard `condition` (`LRRK2`/`Control`) and `timepoint` (`Day0`/`Day10`/`Day14`/`Day42`) columns. Upper-cased variants (`Condition`, `Timepoint`) are also accepted.

### `<gnn-dir>` — pretrained MitoKG
Must contain:
- `gene_embeddings.npy` — (N_genes, 64) HGT-pretrained embeddings
- `gene_list.txt` (or `gene_list`) — gene symbols, one per line

See the `MitoKG` release for these files.

## Quick start

Train on LRRK2:

```bash
python train.py \
    --dataset lrrk2 \
    --data-dir /path/to/lrrk2 \
    --gnn-dir  /path/to/mitokg \
    --out-dir  ./output_lrrk2
```

Train on PINK1:

```bash
python train.py \
    --dataset pink1 \
    --data-dir /path/to/pink1 \
    --gnn-dir  /path/to/mitokg \
    --out-dir  ./output_pink1
```

## Command-line arguments

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | `pink1` or `lrrk2` |
| `--data-dir` | required | directory holding `expr_scaled.npy` and `cell_meta.csv` |
| `--gnn-dir` | required | directory holding `gene_embeddings.npy` and `gene_list.txt` |
| `--out-dir` | `./output` | where to write checkpoints, logs and figures |
| `--epochs` | 500 | number of training epochs |
| `--batch-size` | 256 | mini-batch size |
| `--lr` | 1e-3 | learning rate for the AdamW optimizer |
| `--seed` | 99 | random seed |
| `--mask-ratio` | 0.25 | mitochondrial mask ratio during training |
| `--train-ratio` | 0.70 | train/test split ratio (stratified by timepoint x condition) |
| `--pretrained` | `None` | optional path to a pretrained `.pt` checkpoint (skips training) |

All other hyperparameters (network widths, loss weights, warm-up schedules,
ODE solver settings, etc.) are fixed inside `train.py` to the values used
in the MitoLODE paper.

## Outputs

After a successful run, `<out-dir>` will contain:

```
best_model_<dataset>.pt       best checkpoint by reconstruction loss
best_by_r_<dataset>.pt        best checkpoint by disease Pearson r
last_model_<dataset>.pt       last-epoch checkpoint
training_log.csv              per-epoch metrics
eval_metrics.json             final evaluation metrics
fig1_training.png             training curves
fig2_scatter.png              main-task prediction scatter (disease + control)
fig3_traj_pearson.png         full-trajectory Pearson r per timepoint
fig4_gene_heatmap.png         top-50 gene expression heatmap at final timepoint
fig5_top20_gene_traj.png      top-20 changing genes over time
fig6_pca.png                  PCA trajectory (ground truth vs prediction)
```

## Hardware

Training uses a single GPU. On a single RTX 3090 (24 GB), one full run
takes approximately:

- PINK1 (about 3.9k cells): ~5 minutes for 500 epochs
- LRRK2 (about 36k cells):  ~30 minutes for 500 epochs

Training falls back to CPU if CUDA is unavailable but will be much slower.

## Citation

If you use this code, please cite the MitoLODE paper.
