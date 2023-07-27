.. PhyloVelo documentation master file, created by
   sphinx-quickstart on Thu Aug  4 23:40:12 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PhyloVelo - Phylogeny-based transcriptomic velocity of single cells
=============================================================================

**PhyloVelo** is a computational framework to quantify the transcriptomic velocity field from phylogeny-resolved scRNA-seq data. PhyloVelo utilizes monotonically expressed genes (MEGs) along cell divisions as a transcriptomic clock to reconstruct the velocity vector fields of cell-state transitions, which enables robust and systematic cell-fate mapping in diverse biological contexts.


.. image:: ./diagram_no_text.svg
   :width: 1000px
   :align: center

.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
   
   Installation
   references
   
   
.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:
   
   notebook/getting_start
   notebook/simulation_lineage
   notebook/Simulation_Linear
   notebook/C.elegans_demo
   notebook/KPTracer-3726-t1
   notebook/clone_based_phylovelo

.. toctree::
   :caption: Model Comparsion
   :maxdepth: 1
   :hidden:
   
   notebook/Celegans_model_comparison
   notebook/Erythroid_comparison
   notebook/kp3726_comparison

