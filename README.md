# PhyloVelo

PhyloVelo estimates transcriptomic velocity from phylogeny-resolved single-cell
RNA-seq data. It uses monotonically expressed genes (MEGs) as a transcriptomic
clock, infers gene-wise velocity along cell divisions, and projects the velocity
field onto low-dimensional embeddings for cell-fate and pseudotime analysis.

The method is designed for datasets where lineage or phylogenetic information is
available alongside single-cell expression profiles.

## What PhyloVelo Does

- Identifies MEGs associated with cell generation or lineage time.
- Infers a transcriptomic velocity vector from MEG dynamics.
- Projects velocity into PCA, t-SNE, UMAP, or another 2D embedding.
- Estimates PhyloVelo pseudotime from the velocity field or directly from MEG
  expression.
- Provides plotting utilities for stream, grid, and point velocity fields.

## Installation

```bash
git clone https://github.com/kunwang34/PhyloVelo.git
cd PhyloVelo
pip install .
```

For reproducible environments, install the versions listed in
[`requirements.txt`](requirements.txt).

## Documentation

Full tutorials and examples are available in the
[PhyloVelo documentation](https://phylovelo.readthedocs.io).

Useful starting points:

- [`demo/Simulation_Linear.ipynb`](demo/Simulation_Linear.ipynb)
- [`demo/C.elegans_demo.ipynb`](demo/C.elegans_demo.ipynb)
- [`docs/source/notebook/getting_start.ipynb`](docs/source/notebook/getting_start.ipynb)

## Quick Start

```python
import phylovelo as pv

# count: pandas DataFrame, cells x genes expression matrix
# xdr: pandas DataFrame, cells x 2 embedding coordinates, such as UMAP or t-SNE
# generation: cell generation or lineage time for each cell
sd = pv.scData(
    count=count,
    Xdr=xdr,
    cell_names=list(count.index),
    cell_generation=generation,
)

# Optional but commonly used before inference.
sd.normalize_filter(target_sum=None)

# Infer gene-wise velocity from MEGs.
pv.velocity_inference(
    sd,
    time=sd.cell_generation,
    target="count",
    cutoff=0.95,
)

# Project velocity into the embedding.
pv.velocity_embedding(sd, target="count", n_neigh=100)

# Estimate pseudotime from the embedding velocity graph.
pv.calc_phylo_pseudotime(sd, n_neighbors=100)
```

The main outputs are stored on the `scData` object:

- `sd.megs`: selected monotonically expressed genes.
- `sd.velocity`: inferred gene-wise velocity.
- `sd.velocity_embeded`: velocity vectors projected into the 2D embedding.
- `sd.phylo_pseudotime`: pseudotime normalized to `[0, 1]`.

## Pseudotime Options

PhyloVelo provides two pseudotime estimators.

### Velocity-Graph Pseudotime

This is the original embedding-based strategy. It builds a kNN graph in the
embedding, orients edges using projected velocity, and estimates pseudotime along
a minimum-spanning structure.

```python
pv.calc_phylo_pseudotime(
    sd,
    n_neighbors=100,
    r_sample=0.8,
    random_state=0,
)
```

This implementation is optimized for larger datasets and is robust to disconnected
or widely separated cell clusters.

### MEG Expression Pseudotime

For a more direct and robust estimate, pseudotime can also be inferred from MEG
expression. Each MEG is oriented by its inferred velocity direction, clipped by
per-gene quantiles to reduce outlier influence, aggregated across genes, and
normalized to `[0, 1]`.

```python
pv.calc_meg_pseudotime(sd, target="x_normed")
```

The same estimator can be selected through `calc_phylo_pseudotime`:

```python
pv.calc_phylo_pseudotime(sd, method="meg", target="x_normed")
```

## Plotting

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
pv.velocity_plot(
    sd.Xdr.to_numpy(),
    sd.velocity_embeded,
    ax,
    figtype="stream",
    grid_density=25,
)
```

## Example Data

- **Mouse embryo erythroid**:
  [Google Drive](https://drive.google.com/file/d/1_2SrtJmNILaaKTIoM4VwePOm0_RayrcF/view?usp=drive_link)
- **KP 3726**:
  [Google Drive](https://drive.google.com/file/d/1NsgvD3xylUuZm4CEv5gEeYAciTgFUkov/view?usp=drive_link)

Small demo datasets are included under [`demo/datasets`](demo/datasets).

## Citation

If you use PhyloVelo, please cite:

Kun Wang, Liangzhen Hou, Xin Wang, Xiangwei Zhai, Zhaolian Lu, Zhike Zi,
Weiwei Zhai, Xionglei He, Christina Curtis, Da Zhou, Zheng Hu. PhyloVelo
enhances transcriptomic velocity field mapping using monotonically expressed
genes. *Nature Biotechnology* (2023).
[https://www.nature.com/articles/s41587-023-01887-5](https://www.nature.com/articles/s41587-023-01887-5)

## Development Status

PhyloVelo is under active development. APIs and notebook examples may change
between versions, and results may vary slightly as inference and visualization
utilities are improved.
