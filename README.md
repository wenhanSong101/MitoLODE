# MitoLODE

**MitoLODE: a computational framework for virtual mitochondrial research. Using a mitochondrial knowledge graph as a structural prior and a neural ordinary differential equation as the dynamics backbone, we model the continuous-time behavior of mitochondria at single-cell resolution under both healthy and disease states, and combine large language models with independent database validation to provide mechanistic-level interpretation of perturbation targets.**

This repository provides the reference implementation that accompanies the MitoLODE paper. It contains: the mitochondrial knowledge graph MitoKG, the MitoLODE model and its training code, forward and reverse virtual perturbation experiments, the MGMI (MitoKG-Grounded Mechanistic Inference) interpretation module, and a publication-quality MitoKG visualization script.

## Repository layout

```
MitoLODE/
├── MitoKG/                    pretrained mitochondrial heterogeneous knowledge graph
│   ├── gnn/                   hetero_graph.pt, gene_embeddings.npy, ...
│   └── README.md
├── data/
│   └── lrrk2/                 processed LRRK2 dataset (see "Data availability")
├── mitolode/                  main model code
│   ├── model.py               MitoLODE network architecture
│   ├── train.py               training entry point
│   └── README.md
├── perturbation/              virtual perturbation experiments
│   ├── perturbation_forward.py  forward perturbation (Disease -> Control)
│   ├── perturbation_reverse.py  reverse perturbation (Control -> Disease)
│   └── README.md
├── mgmi/                      MitoKG-grounded LLM mechanistic inference
│   ├── mgmi_inference.py        DeepSeek reasoning-chain generation
│   ├── mgmi_external_validation.py  OpenTargets + PubMed + Ensembl validation
│   └── README.md
├── figures/                   publication-quality graph visualization
│   ├── mitokg_visualize.py
│   └── README.md
├── requirements.txt
├── LICENSE                    MIT License for the code
└── README.md                  this file
```

## Quick start

```bash
# 1. install dependencies
pip install -r requirements.txt

# 2. train MitoLODE on the LRRK2 dataset (default seed = 99)
python mitolode/train.py \
    --dataset lrrk2 \
    --data-dir   ./data/lrrk2 \
    --gnn-dir    ./MitoKG/gnn \
    --out-dir    ./output_lrrk2

# 3. forward perturbation (Disease -> Control)
python perturbation/perturbation_forward.py \
    --data-dir ./data/lrrk2 \
    --gnn-dir  ./MitoKG/gnn \
    --weight   ./output_lrrk2/best_by_r_lrrk2.pt \
    --out-dir  ./output_perturbation_forward

# 4. MGMI reasoning chains (you must provide your own DeepSeek API key)
export DEEPSEEK_API_KEY="sk-your-key-here"
python mgmi/mgmi_inference.py \
    --gnn-dir      ./MitoKG/gnn \
    --perturb-xlsx ./output_perturbation_forward/perturbation_lrrk2_temporal.xlsx \
    --out-dir      ./output_mgmi

# 5. independent external validation
python mgmi/mgmi_external_validation.py \
    --mgmi-json ./output_mgmi/mitokg_llm_raw_v2.json \
    --out-dir   ./output_mgmi_validation
```

Each subdirectory has a dedicated `README.md` with full usage details.

## What this repository releases

- **MitoKG**: the full pretrained knowledge graph (1132 genes, 815 pathways, 305 drugs, 297 complexes, 23019 edges), including 64-dimensional HGT embeddings. Data sources and construction details are documented in `MitoKG/README.md`.
- **MitoLODE main model** (`mitolode/`): the network architecture, training loop, and evaluation code.
- **Virtual perturbation** (`perturbation/`): forward (Dis -> Ctl) and reverse (Ctl -> Dis) four-layer perturbation experiments.
- **MGMI** (`mgmi/`): LLM-based mechanistic reasoning chains, plus external validation against OpenTargets, PubMed, and Ensembl.
- **MitoKG visualization** (`figures/`): the script that produces Figure 1 of the paper.

## Currently being prepared for release

The following will be added to this repository on a rolling basis alongside the paper:

- **Hold-out time-point extrapolation** code.
- **Single-cell data processing pipeline** (raw GEO data -> whitelist gene alignment -> normalization -> the `expr_scaled.npy` and `cell_meta.csv` files consumed directly by MitoLODE).

## Data availability

### MitoKG

MitoKG is mirrored on Zenodo with a permanent DOI under the CC BY 4.0 license.

> **DOI**: `10.5281/zenodo.19688390`
>
> The contents of the `MitoKG/` directory in this repository are identical to the Zenodo release; please cite the DOI above.

### Processed single-cell data

The paper uses four publicly available iPSC differentiation single-cell transcriptomic datasets. **The processed expression matrices and the corresponding processing pipeline (raw GEO data -> whitelist gene alignment -> normalization -> the `expr_scaled.npy` and `cell_meta.csv` files consumed directly by MitoLODE) are currently being cleaned up and will be uploaded to this repository on a rolling basis.**

The original data are publicly available from NCBI GEO under the following accessions:

| Dataset              | GEO accession  | Original publication                                                                                                                                    |
|----------------------|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| LRRK2                | GSE128040      | Walter et al., *Stem Cell Reports* 12(5):878-889, 2019 (LRRK2-G2019S iPSC differentiation to dopaminergic neurons; Day 0 / 10 / 14 / 42)                |
| PINK1                | GSE183248      | Novak et al., *Commun Biol* 5(1):49, 2022 (PINK1-ILE368ASN and control iPSC differentiation to midbrain dopaminergic neurons; D6 / D15 / D21)       |
| Cardiac              | GSE175634      | Elorbany, Popp et al., *PLOS Genetics* 18(1):e1009666, 2022 (iPSC differentiation to cardiomyocytes, 7 time points)                                     |
| Pancreatic beta-cell | GSE114412      | Veres et al., *Nature* 569:368-373, 2019 (iPSC differentiation to pancreatic endocrine cells)                                                            |

The processing scripts will be added as they are cleaned up.

## Reproducibility

- All random seeds default to `99`.
- Hyperparameters in `mitolode/` and `perturbation/` are hard-coded to the LRRK2 configuration reported in the paper; once the data and MitoKG are in place, `python train.py` followed by `python perturbation_forward.py` reproduces the paper's numbers up to small numerical noise (chiefly due to non-determinism in GPU scatter-gather operations at the reported precision).
- The MGMI reasoning step uses the DeepSeek Chat API with `temperature=0.3`, but the API itself is non-deterministic server-side. The reasoning chains reported in the paper were produced against the `deepseek-chat` model version served during the experiment.
- The external validation step queries three live public APIs, so small drift between runs is expected over time.

## License

- **Code** (`mitolode/`, `perturbation/`, `mgmi/`, `figures/`): MIT License. See `LICENSE`.
- **MitoKG data** (everything under `MitoKG/`): CC BY 4.0. Attribution to the five upstream sources (STRING, Reactome, DrugBank, CORUM, MitoCarta) is documented in `MitoKG/README.md`.
