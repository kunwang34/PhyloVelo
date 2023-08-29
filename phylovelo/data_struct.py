from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import issparse

from .ana_utils import *


class scData:
    '''
    Data structure for PhyloVelo analysis
    
    Args:
        count: Read/UMI count. Index: cell names, columns: gene names
        x_normed: Normalized count. Index: cell names, columns: gene names
        latent_z: Inferenced latent expression
        Xdr: PCA/UMAP or tSNE coordinate, n cells * 2
        phylo_tree: Phylogenetic tree
        cell_states: Cell types
        cell_names: Same to count's index
        cell_generation: Generation time of cells
        megs: MEGs
        velocity: PhyloVelo velocity
        velocity_embeded: PhyloVelo velocity project into embedding
        phylo_pseudotime: Pseudotime inferenced by PhyloVelo

    '''
    def __init__(
        self,
        count: "pandas DataFrame" = None,
        x_normed: "pandas DataFrame" = None,
        latent_z: "pandas DataFrame" = None,
        Xdr: "pandas DataFrame" = None,
        phylo_tree: "phylo.tree" = None,
        cell_states: list = None,
        cell_names: list = None,
        cell_generation: list = None,
        megs: list = None,
        velocity: list = None,
        velocity_embeded: list = None,
        phylo_pseudotime: list = None,
        pvals: list = None,
        qvals: list = None
    ):
        self.count = count
        self.x_normed = x_normed
        self.latent_z = latent_z
        self.Xdr = Xdr
        self.cell_names = cell_names
        self.phylo_tree = phylo_tree
        self.cell_states = cell_states
        self.cell_generation = cell_generation
        self.megs = megs
        self.velocity = velocity
        self.velocity_embeded = velocity_embeded
        self.phylo_pseudotime = phylo_pseudotime
        self.pvals = pvals
        self.qvals = qvals

    def drop_duplicate_genes(self, target="count"):
        '''
        Remove duplicated genes
        
        Args:
            target: count or x_normed
        '''
        dup_genes = set()
        genes = set()
        data = getattr(self, target)
        for i in data.columns:
            if i in genes:
                dup_genes.add(i)
            genes.add(i)
        try:
            self.count = self.count.drop(dup_genes, axis=1)
        except:
            None
        try:
            self.x_normed = self.x_normed.drop(dup_genes, axis=1)
        except:
            None
            
    def normalize_filter(
        self, is_normalize=True, is_log=True, min_count=10, target_sum=None
    ):
        '''
        normalize read/umi count and filter genes
        
        Args:
            is_normalize: Similiar to normalize_total in scanpy. True for normalize
            is_log: log(1+X)
            min_count: filter genes total count < min_count
            target_sum: if None, use median
        
        Return:
            self.x_normed
        '''
        if min_count:
            self.count = self.count.loc[:, self.count.sum(axis=0) > min_count]

        if is_normalize:
            counts_per_cell = self.count.sum(axis=1)
            target_sum = (
                np.median(counts_per_cell[counts_per_cell > 0], axis=0)
                if target_sum is None
                else target_sum
            )
            counts_per_cell += counts_per_cell == 0
            counts_per_cell = counts_per_cell / target_sum
            self.x_normed = np.divide(self.count, counts_per_cell.to_numpy().reshape(-1,1))

        if is_log:
            if self.x_normed is None:
                self.x_normed = self.count.copy()
            self.x_normed = np.log(1 + self.x_normed)

    def dimensionality_reduction(
        self,
        target: "count, x_normed" = "count",
        method: "pca, tsne, umap" = "tsne",
        n_components: int = 2,
        scale:float=1,
        pc:bool=True,
        **kwags
    ):
        '''
        PCA/tSNE or UMAP
        
        Args:
            target: count 
            method: use PCA/tSNE or UMAP
            n_components: Reduce the dimension to 'n_components'
            scale: normalize scale
            pc: Use PCA to tSNE/UMAP or not
            pc_components: How many PC to use when tSNE/UMAP
            perplexity: tSNE perplexity
            n_neighbors: UMAP n_neighbors
            min_dist: UMAP min_dist
            
        Return:
            self.Xdr
        '''

        assert method in ["pca", "tsne", "umap"]

        pc_components = kwags.pop("pc_components", 100)
        perplexity = kwags.pop("perplexity", 80)
        n_neighbors = kwags.pop("n_neighbors", 15)
        min_dist = kwags.pop("min_dist", 0.5)

        self.dimr_method = method

        if target == "x_normed":
            X_norm = self.x_normed
        else:
            X_norm = logNormalize(self.count, scale)

        X_pc100 = PCA(n_components=pc_components).fit_transform(X_norm)

        if method == "pca":
            self.dim_reducer = PCA(n_components=n_components)

        elif method == "tsne":
            self.dim_reducer = TSNE(n_components=n_components, perplexity=perplexity)

        else:
            import umap

            self.dim_reducer = umap.UMAP(
                n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
            )
        if pc:
            Xdr = self.dim_reducer.fit_transform(X_pc100)
        else:
            Xdr = self.dim_reducer.fit_transform(X_norm)
        self.Xdr = pd.DataFrame(Xdr, index=self.cell_names)
