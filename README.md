# PhyloVelo - Phylogeny-based transcriptomic velocity of single cells

PhyloVelo is a computational framework to quantify the transcriptomic velocity field from phylogeny-resolved scRNA-seq data. PhyloVelo utilizes monotonically expressed genes (MEGs) along cell divisions as a transcriptomic clock to reconstruct the velocity vector fields of cell-state transitions, which enables robust and systematic cell-fate mapping in diverse biological contexts.

## Installation
```
git clone https://github.com/kunwang34/PhyloVelo
cd ./PhyloVelo
pip install .
```

## Docs
[PhyloVelo Docs](https://phylovelo.readthedocs.io)

## References
Kun Wang, Liangzhen Hou, Xin Wang, Xiangwei Zhai, Zhaolian Lu, Zhike Zi, Weiwei Zhai, Xionglei He, Christina Curtis, Da Zhou\*, Zheng Hu\*. PhyloVelo for robust mapping of single-cell velocity fields using monotonically expressed genes. hyperlink/doi (2023).(to be updated)

Bergen, V., Lange, M., Peidli, S., Wolf, F.A. & Theis, F.J. Generalizing RNA velocity to transient cell states through dynamical modeling. Nat Biotechnol. https://doi.org/10.1038/s41587-020-0591-3

Gao, M., Qiao, C. & Huang, Y. UniTVelo: temporally unified RNA velocity reinforces single-cell trajectory inference. Nat Commun. https://doi.org/10.1038/s41467-022-34188-7

Shengyu Li, Pengzhi Zhang, Weiqing Chen, Lingqun Ye, Kristopher W. Brannan, Nhat-Tu Le, Jun-ichi Abe, John P. Cooke, Guangyu Wang. A relay velocity model infers cell-dependent RNA velocity. Nature Biotechnology. https://doi.org/10.1038/s41587-023-01728-5

Haotian Cui, Hassaan Maan, Michael D. Taylor, Bo Wang. DeepVelo - A Deep Learning-based velocity estimation tool with cell-specific kinetic rates. GitHub. https://github.com/bowang-lab/DeepVelo

Yichen Gu, David Blaauw,  Joshua D. Welch. VeloVAE - Variational Mixtures of ODEs for Inferring Cellular Gene Expression Dynamics. GitHub. https://github.com/welch-lab/VeloVAE

## Note
PhyloVelo is still under development, some API names and call methods in notebooks may have changed, and the model performance may also be slightly different.
