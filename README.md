# PhyloVelo - Phylogeny-based transcriptomic velocity of single cells

PhyloVelo is a computational framework to quantify the transcriptomic velocity field from phylogeny-resolved scRNA-seq data. PhyloVelo utilizes monotonically expressed genes (MEGs) along cell divisions as a transcriptomic clock to reconstruct the velocity vector fields of cell-state transitions, which enables robust and systematic cell-fate mapping in diverse biological contexts.

## Reference
Kun Wang, Liangzhen Hou, Xin Wang, Xiangwei Zhai, Zhaolian Lu, Zhike Zi, Weiwei Zhai, Xionglei He, Christina Curtis, Da Zhou\*, Zheng Hu\*. A refined velocity model using monotonically expressed genes improves cell fate mapping. hyperlink/doi (2023).(to be updated)

## Installation
```
git clone https://github.com/kunwang34/PhyloVelo
cd ./PhyloVelo
pip install .
```

## Docs
[PhyloVelo Docs](https://phylovelo.readthedocs.io)

## Note
PhyloVelo is still under development, some API names and call methods in notebooks may have changed, and the model performance may also be slightly different.
