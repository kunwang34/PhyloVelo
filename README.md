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


## Data

- **Mouse embryo erythroid**
  
  https://drive.google.com/file/d/1_2SrtJmNILaaKTIoM4VwePOm0_RayrcF/view?usp=drive_link
- **KP 3726**
  
  https://drive.google.com/file/d/1NsgvD3xylUuZm4CEv5gEeYAciTgFUkov/view?usp=drive_link

## References
Kun Wang, Liangzhen Hou, Xin Wang, Xiangwei Zhai, Zhaolian Lu, Zhike Zi, Weiwei Zhai, Xionglei He, Christina Curtis, Da Zhou\*, Zheng Hu\*. PhyloVelo enhances transcriptomic velocity field mapping using monotonically expressed genes. Nature Biotechnology. https://www.nature.com/articles/s41587-023-01887-5 (2023.07).

## Note
PhyloVelo is still under development, some API names and call methods in notebooks may have changed, and the model performance may also be slightly different.
