Pseudotime
==========

PhyloVelo provides two ways to calculate pseudotime. Both methods write the
result to ``sd.phylo_pseudotime`` and scale it to the interval ``[0, 1]``.

Velocity-graph pseudotime
-------------------------

The default method estimates pseudotime from the projected velocity field in a
two-dimensional embedding. It builds a k-nearest-neighbor graph over
``sd.Xdr``, orients local time intervals using ``sd.velocity_embeded``, and then
propagates time along a minimum-spanning structure.

.. code-block:: python

   pv.velocity_embedding(sd, target="count", n_neigh=100)
   pv.calc_phylo_pseudotime(sd, n_neighbors=100)

For larger datasets, a random subset can be used to estimate the graph
pseudotime and then interpolate values back to all cells:

.. code-block:: python

   pv.calc_phylo_pseudotime(
       sd,
       n_neighbors=100,
       r_sample=0.8,
       random_state=0,
   )

The current implementation performs one batched nearest-neighbor search and uses
a heap-based minimum-spanning-tree routine. If the kNN graph contains isolated
points or separated cell populations, PhyloVelo connects the components with
nearest bridging edges instead of failing.

Recommended use
~~~~~~~~~~~~~~~

Use velocity-graph pseudotime when the embedding and projected velocity field
are reliable and you want pseudotime to reflect the inferred flow direction in
the cell-state manifold.

Important parameters
~~~~~~~~~~~~~~~~~~~~

``n_neighbors``
   Number of embedding neighbors used to build the graph. Larger values make
   disconnected components less likely, but can smooth over local structure.

``r_sample``
   Fraction of cells sampled for graph construction. Use values below ``1`` for
   large datasets.

``random_state``
   Seed used when ``r_sample < 1``.

MEG expression pseudotime
-------------------------

PhyloVelo can also estimate pseudotime directly from monotonically expressed
genes (MEGs). This method orients each MEG by the sign of its inferred velocity,
clips expression values by per-gene quantiles to reduce outlier influence,
aggregates the oriented gene-wise scores, and scales the result to ``[0, 1]``.

.. code-block:: python

   pv.velocity_inference(sd, time=sd.cell_generation, target="x_normed")
   pv.calc_meg_pseudotime(sd, target="x_normed")

The same estimator can be selected through ``calc_phylo_pseudotime``:

.. code-block:: python

   pv.calc_phylo_pseudotime(sd, method="meg", target="x_normed")

Recommended use
~~~~~~~~~~~~~~~

Use MEG expression pseudotime when you want a direct transcriptomic-clock score,
when the embedding has separated clusters, or when the velocity graph is too
slow or too sensitive to the manifold geometry.

Important parameters
~~~~~~~~~~~~~~~~~~~~

``target``
   Expression matrix used for scoring, usually ``"x_normed"`` or ``"count"``.

``genes``
   Optional list of MEGs. By default, PhyloVelo uses ``sd.megs`` from
   ``velocity_inference``.

``robust_quantiles``
   Lower and upper quantiles used to clip each gene before scaling. The default
   is ``(0.05, 0.95)``.

``aggregation``
   ``"median"`` is the default and is robust to noisy genes. ``"weighted_mean"``
   uses the absolute inferred velocity as a gene weight.

Choosing a method
-----------------

.. list-table::
   :header-rows: 1

   * - Situation
     - Suggested method
   * - Reliable continuous embedding and velocity field
     - ``calc_phylo_pseudotime(sd, method="graph")``
   * - Large datasets where graph construction is expensive
     - Graph method with ``r_sample < 1`` or MEG expression pseudotime
   * - Separated clusters or isolated cells
     - MEG expression pseudotime, or graph method with enough neighbors
   * - Direct clock-like ordering from MEGs
     - ``calc_meg_pseudotime(sd)``

Notes
-----

- ``sd.phylo_pseudotime`` is always normalized to ``[0, 1]``.
- The graph method requires ``sd.Xdr`` and ``sd.velocity_embeded``.
- The MEG expression method requires MEGs and velocity directions, typically
  produced by ``velocity_inference``.
