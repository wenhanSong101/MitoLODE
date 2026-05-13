# MitoKG — Mitochondrial Heterogeneous Knowledge Graph

A pretrained heterogeneous knowledge graph centered on mitochondrial biology,
integrating gene–gene interactions, pathway hierarchies, drug–target relations,
and protein complexes. Released together with **64-dimensional HGT-pretrained node
embeddings** that can be used directly as biological priors in downstream models.

This resource is part of the **MitoLODE** project.

---

## 1. Overview

MitoKG is a four-node-type heterogeneous graph built around the human mitochondrial
proteome. The graph was constructed by intersecting the MitoCarta 3.0 core
mitochondrial gene list with five public biomedical databases, mapping every
identifier to UniProt, and assembling typed edges between four node categories:
**gene/protein, pathway, drug, protein complex**.

The graph was then pretrained with a **Heterogeneous Graph Transformer (HGT)** to
produce 64-dim node embeddings that encode multi-relational biological context.
These embeddings serve as the core biological prior for MitoLODE, and can be
reused in any downstream model that needs mitochondria-aware gene features.

Released artifacts include: the `HeteroData` graph object, HGT model weights,
node embeddings (both the 984-gene core subset and the full 1,132-gene set),
a UniProt-to-symbol mapping of the 984 core genes, and node-ID lookup tables.

---

## 2. Graph Statistics

Exact counts read directly from the released `hetero_graph.pt`:

### 2.1 Nodes (total = 2,549)

| Node type | Count | Raw feature dim | Description |
|---|---|---|---|
| `gene`    | **1,132** | 57 | Genes/proteins connected to the mitochondrial core via PPI, pathway or complex membership |
| `pathway` | **815**   | 21 | Reactome + MitoCarta pathway hierarchy |
| `drug`    | **305**   | 20 | DrugBank entries targeting at least one gene in the graph (210 approved) |
| `complex` | **297**   | 2  | CORUM protein complexes (65 annotated as core mitochondrial) |

### 2.2 Edges (forward, total = 23,019)

| Edge type | Count | Source |
|---|---|---|
| `(gene, interacts, gene)`       | **14,763** | STRING v12.0, high-confidence PPI (score ≥ 700, *Homo sapiens* 9606) |
| `(gene, involved_in, pathway)`  | **5,990**  | Reactome (UniProt2Reactome_All_Levels) |
| `(gene, member_of, complex)`    | **806**    | CORUM 5.2 |
| `(pathway, has_child, pathway)` | **804**    | Reactome hierarchy |
| `(drug, targets, gene)`         | **656**    | DrugBank (latest full database) |

The graph also stores reverse edges for all non-symmetric relations (rev_involved_in,
rev_targets, rev_member_of, rev_has_child) so that messages can propagate in both
directions during HGT training.

### 2.3 Coverage (from the 984 core mitochondrial gene list)

- Proteins with at least one PPI edge: **970 / 984**
- Proteins with at least one pathway annotation: **860 / 984**
- Proteins with at least one drug targeting: **292 / 984**
- Proteins with at least one complex membership: **345 / 984**

---

## 3. Data Sources

All five databases are used under their own respective licenses.
MitoKG only distributes derived objects; users must obtain the original
source databases from the official links below.

| Database | Version used | URL | Purpose |
|---|---|---|---|
| **MitoCarta** | 3.0 | https://www.broadinstitute.org/mitocarta | Mitochondrial gene whitelist (seed set) and pathway annotations |
| **UniProt**   | REST API (queried 2026) | https://www.uniprot.org | Canonical ID mapping (symbol ↔ UniProt accession) |
| **STRING**    | v12.0 (*Homo sapiens* 9606) | https://string-db.org | Protein–protein interactions (score ≥ 700) |
| **Reactome**  | current (UniProt2Reactome_All_Levels) | https://reactome.org | Pathway membership + pathway hierarchy |
| **DrugBank**  | latest full database | https://go.drugbank.com | Drug–target relations (annotated as approved / investigational) |
| **CORUM**     | 5.2 (release 2025-11-07) | https://mips.helmholtz-muenchen.de/corum/ | Curated protein complexes |

Please cite each original database according to its own guidance if you use
derived MitoKG artifacts in your work. The authoritative data licences remain
with the original providers.

---

## 4. Construction Pipeline (summary)

1. **Seed set**: Start from the 1,136-gene human MitoCarta 3.0 list.
2. **UniProt mapping**: Map every gene symbol to canonical UniProt accession;
   drop entries without an unambiguous human UniProt ID. This yields the
   **984-gene core mitochondrial whitelist** released here as `gene_list.txt`.
3. **Edge harvesting**: For each of the 984 seed proteins, pull
   - high-confidence STRING PPI partners (score ≥ 700),
   - Reactome pathway memberships and the pathway hierarchy above them,
   - DrugBank drugs targeting the protein,
   - CORUM complexes it participates in.
4. **Graph expansion**: Add any gene/protein that appears on the partner side of
   a retained edge. This expands the gene-node set from 984 seeds to the
   **final 1,132 gene nodes** in the graph.
5. **Feature assembly**: Compute raw node features per type
   (gene: 57-dim — structural/functional annotations; pathway: 21-dim; drug: 20-dim;
   complex: 2-dim). Used as HGT input.
6. **HGT pretraining**: Train a Heterogeneous Graph Transformer over the assembled
   graph with a self-supervised link-prediction objective; export node embeddings
   at hidden dim 64.

---

## 5. Released Files

```
mitokg/
├── README.md                     this file
├── gene_list.txt                 984 gene symbols (one per line) — the core whitelist
├── hetero_graph.pt               PyG HeteroData object: full graph with 4 node types and 5 forward + 4 reverse edge types
├── hgt_model.pt                  trained HGT weights + metadata (hidden_dim=64, 176,720 parameters, 70 tensors)
├── gene_embeddings.npy           (984, 64) float32 — HGT embeddings for the 984 core mitochondrial genes
├── all_gene_embeddings.npy       (1132, 64) float32 — HGT embeddings for every gene node in the graph
├── node_id_maps.pkl              dict of {node_type: {name → index}} — maps back from embedding rows to biological identifiers
└── graph_stats.txt               plain-text summary of graph construction
```

### File details

| File | Format | Shape / Type | Description |
|---|---|---|---|
| `gene_list.txt`            | plain text | 984 lines       | Gene symbols in the same order as rows of `gene_embeddings.npy`. |
| `hetero_graph.pt`          | PyTorch    | `HeteroData`    | Load with `torch.load(..., weights_only=False)`. Contains `.x` features for each node type and `.edge_index` tensors for each edge type. |
| `hgt_model.pt`             | PyTorch    | dict            | Keys: `model_state` (state_dict), `hidden_dim` (int, = 64), `metadata` (PyG `metadata()` tuple describing node/edge types). |
| `gene_embeddings.npy`      | NumPy      | (984, 64) f32   | Row *i* corresponds to line *i* of `gene_list.txt`. |
| `all_gene_embeddings.npy`  | NumPy      | (1132, 64) f32  | Row *i* corresponds to gene node *i* in `hetero_graph.pt`. Use `node_id_maps.pkl` to resolve which gene is which row. |
| `node_id_maps.pkl`         | pickle     | nested dict     | `{'gene': {symbol: idx}, 'pathway': {name: idx}, 'drug': {name: idx}, 'complex': {name: idx}}`. Loaded with `pickle.load()`. |
| `graph_stats.txt`          | plain text | report          | Human-readable snapshot of graph construction. |

---

## 6. Quick Start

### 6.1 Use the pretrained gene embeddings (simplest)

Drop-in mitochondria-aware gene features for any downstream model:

```python
import numpy as np

# 984-core version — matches MitoLODE's input feature set
emb = np.load('gene_embeddings.npy')         # (984, 64)
with open('gene_list.txt') as f:
    gene_symbols = [line.strip() for line in f]

# Example: get embedding for PINK1
idx = gene_symbols.index('PINK1')
pink1_vec = emb[idx]                          # (64,)
```

### 6.2 Load the full heterogeneous graph

```python
import torch
from torch_geometric.data import HeteroData

data = torch.load('hetero_graph.pt', weights_only=False)
print(data)
# HeteroData(
#   gene={x=[1132, 57]},
#   pathway={x=[815, 21]},
#   drug={x=[305, 20]},
#   complex={x=[297, 2]},
#   (gene, interacts, gene)={edge_index=[2, 14763], edge_weight=[14763]},
#   (gene, involved_in, pathway)={edge_index=[2, 5990]},
#   ...
# )

# All gene-gene PPI edges
ppi = data['gene', 'interacts', 'gene'].edge_index          # (2, 14763)
ppi_w = data['gene', 'interacts', 'gene'].edge_weight       # (14763,)
```

### 6.3 Reload the HGT and re-embed

```python
import torch
from torch_geometric.nn import HGTConv

ckpt = torch.load('hgt_model.pt', weights_only=False, map_location='cpu')
print('hidden_dim:', ckpt['hidden_dim'])          # 64
print('metadata:',   ckpt['metadata'])            # PyG metadata tuple

# You will need to instantiate a matching HGT module and call
# model.load_state_dict(ckpt['model_state']) before running inference.
# Architecture: 4 node types × 5 forward edge types, 2 HGT layers, 64-dim hidden.
```

### 6.4 Map rows back to genes

```python
import pickle, numpy as np

id_maps = pickle.load(open('node_id_maps.pkl', 'rb'))
emb_all = np.load('all_gene_embeddings.npy')       # (1132, 64)

# What's the embedding of MT-CO1?
idx = id_maps['gene']['MT-CO1']
vec = emb_all[idx]
```

---

## 7. Relation to the MitoLODE Paper

In MitoLODE, these embeddings serve as the **mitochondria-specific biological
prior** that enters three separate components of the model:

- **KGSFM encoder gate**: gene-level modulation of the VAE encoder.
- **KG-ODE dynamics**: context vector injected into the ODE velocity field.
- **KG-Decoder**: gene-level skip connection to the output layer.

Fine-grained ablations isolating each component are reported in the paper.
If you only want the biological prior, use `gene_embeddings.npy` directly.

---

## 8. Licence and Attribution

- **MitoKG-derived artifacts** in this directory (graph object, model weights,
  embeddings, node-ID maps, gene list) are released for academic and research
  use. If you build on them, please cite the MitoLODE paper.
- **Source databases retain their original licences.** MitoKG never redistributes
  raw MitoCarta / STRING / Reactome / DrugBank / CORUM tables. Users who want to
  rebuild or extend the graph must download the original sources themselves under
  the respective databases' terms.
- The MitoCarta 3.0 human gene list is a product of the Broad Institute; any
  use of MitoCarta-derived information in downstream publications should cite
  Rath *et al.*, *Nucleic Acids Research* 2021.

---

## 9. Versioning

| Aspect | Value |
|---|---|
| MitoKG snapshot | v1.0 (matches MitoLODE paper) |
| HGT hidden dim | 64 |
| HGT total parameters | 176,720 |
| Graph construction date | 2026 |

Future versions will bump this number and preserve old snapshots in separate
release tags.

---

## 10. Citation

```bibtex
@article{mitolode2026,
  title   = {MitoLODE: A Virtual Mitochondrion via Knowledge-Graph-Informed Neural ODE},
  author  = {...},
  journal = {Briefings in Bioinformatics},
  year    = {2026},
  note    = {Pretrained knowledge graph MitoKG released alongside the paper.}
}
```

(Replace the placeholder once the paper is accepted.)
