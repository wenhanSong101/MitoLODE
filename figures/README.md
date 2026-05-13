# MitoKG Visualization

This folder contains the script that produces the **MitoKG graph
visualization** used as Figure 1 in the paper.

| Script                  | Purpose                                                  |
|-------------------------|----------------------------------------------------------|
| `mitokg_visualize.py`   | Render the MitoKG heterogeneous graph as a full-graph figure, with four node types and five edge types colored and laid out for publication. |

## Input

The script reads directly from the released MitoKG `gnn/` directory:

- `hetero_graph.pt` &mdash; PyG `HeteroData` with all nodes and edges.
- `node_id_maps.pkl` &mdash; node-index to biological identifier maps.
- `gene_list.txt` &mdash; gene symbols in the graph's internal order.

No extra tabular files (CSV/XLSX) are needed.

## Usage

Preview mode (~600 nodes, 3-8 minutes, smaller figure):

```bash
python figures/mitokg_visualize.py \
    --gnn-dir ./MitoKG/gnn \
    --out-dir ./output_visualization
```

Full-graph mode (2549 nodes, 23019 edges, 20-35 minutes, larger figure):

```bash
python figures/mitokg_visualize.py \
    --gnn-dir ./MitoKG/gnn \
    --out-dir ./output_visualization \
    --full
```

## Output

Three formats are produced in `--out-dir`:

- `mitokg_visualization.png` (300 DPI)
- `mitokg_visualization.svg`
- `mitokg_visualization.pdf`

## Visual encoding

- **Node color** (sequential colormap, intensity = degree):
  gene = Blues, complex = GnBu, pathway = YlOrRd, drug = RdPu.
- **Node size** = non-linear function of node degree.
- **Edge color**: PPI = blue, gene-pathway = orange, drug-target =
  magenta, protein-complex = green, pathway-hierarchy = purple.
- **Edge shape**: interior edges are drawn as arcs, peripheral edges as
  straight lines, to reduce visual clutter near the rim.
- **Layout**: two-pass `networkx.spring_layout`, ellipse clipping, and a
  random scatter applied only to peripheral pathway nodes to break up
  the "ring" artifact that a naive force layout produces.

## Reproducibility

The random seed is controlled by `--seed` (default `42`). The same seed
deterministically reproduces the same layout.

## Runtime notes

Full-graph rendering is memory- and time-intensive because 14,763 PPI
edges are drawn as individual patches. Start with the preview mode to
confirm the pipeline works before committing to the full run.
