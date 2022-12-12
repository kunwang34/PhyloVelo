import re
import warnings
from copy import deepcopy
from io import StringIO
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import Phylo
from scipy.stats import chi2, nbinom, pearsonr
from sklearn.neighbors import NearestNeighbors


'''
def highly_variable_genes(data: "pd.DataFrame", n_top_genes:int=2000):
    """
    Useless, may remove.
    
    Args:
        data (pandas.DataFrame): 
            Gene expresison count data
        n_top_genes (int): 
            Number of highly variable genes
        
    Returns:
        pandas.core.series.Series
    """
    import scanpy as sc

    adata = sc.AnnData(data)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")
    return adata.var["highly_variable"]

def get_sisters(tree:'Bio.Phylo.Tree', target:'Bio.Phylo.Clade'):
    """
    Get sisters from phylogenetic tree.
    
    Args:
        tree (Bio.Phylo.Tree): 
            Tree object from Biopython
        target (Bio.Phylo.Clade): 
            Clade object from Biopython
        
    Returns:
        list[str]: 
            Sisters' name of given target
        list[int]: 
            Distances from sisters to target
        int: 
            Distances from common ancestor to root
    """
    path = tree.get_path(target)
    for anc in path[::-1][1:]:
        if len(anc.get_terminals()) > 1:
            return (
                [i.name for i in anc.get_terminals()],
                [tree.distance(anc, i) for i in anc.get_terminals()],
                tree.distance(tree.root, anc),
            )

def get_common_ancestor2(trans_mat, target1, target2):
    """
    Get common ancestor of two
    return: ancestors' state of target1 and target2
    """
    ancs = [target1]
    while not trans_mat[target1] == target1:
        target1 = trans_mat[target1]
        ancs.append(target1)
    while not target2 in ancs:
        target2 = trans_mat[target2]
    return target2


def get_common_ancestor(trans_mat, targets):
    """ """
    while len(targets) > 1:
        target1 = targets.pop()
        target2 = targets.pop()
        targets.append(get_common_ancestor2(trans_mat, target1, target2))
    return targets[0]
'''

def loadtree(file):
    '''
    Reformat tree file from simulation data
    
    Args:
        file(str): 
            File path generated by simulation code
        
    Returns:
        Bio.Phylo.Tree: 
            biopython's phylo tree
        list[str]: 
            cell types of leave nodes
    '''
    with open(file) as f:
        nwt = f.readline()
        nwt = re.sub(">:[0-9].[0-9]*[\)]", ">:1)", nwt)
        nwt = re.sub(">:[0-9].[0-9]*[,]", ">:1,", nwt)
        nwt = re.sub(">:[0-9].[0-9]*[\n]", ">:1\n", nwt)
        tree = Phylo.read(StringIO(nwt), "newick")
        colors = f.readline()
        colors = colors.split("\t")[:-1]
        return tree, colors


def logNormalize(data, scaling=1):
    '''
    Log normalize data
    
    Arg:
        data(pandas.DataFrame, numpy.array): 
            expression data
        scaling(int): 
            Normalization scale
    
    Return:
        normalized data
    '''
    return np.log(1 + data / (np.sum(data, axis=0) + 1e-9) * scaling)


def plot_tree(tree:'Bio.Phylo.tree', colors, ax, stain="all"):
    """
    Plot the simulated phylogenetic tree.
    
    Args:
        tree (Bio.Phylo.tree): 
            Loaded from loadtree function
        ax (matplotlib.axes): 
            Axes to return tree plot
        colors (list): 
            Loaded from loadtree
        is_show (bool): 
            Show the figure or return an axes
        stain ('all', 'terminals'): 
            'all': color all branches; 'terminals': color leave only

    Return:
        matplotlib.axes
    """
    colortab = ["gray", "blue", "green", "orange", "red"]
    names = [i[:-2] for i in colors]
    states = [int(i[-1]) for i in colors]
    states = dict(zip(names, states))
    for i in colors:
        tree.common_ancestor({"name": i[:-2]}).color = colortab[int(i[-1])]

    def cid_ext(x):
        x = str(x.name)
        return

    if stain == "terminals":
        for c in tree.get_nonterminals():
            c.color = "#cdcdcd"
    fig, ax = plt.subplots()
    Phylo.draw(tree, label_func=cid_ext, do_show=is_show, axes=ax)
    return ax


def get_weight(x:list, distance:list, scale, length: int):
    '''
    Weight sum the velocity to grid
    
    Args:
        x: 
            neighbors
        distance: 
            List of distance to neighbors
        scale: 
            Scale factor
        length: 
            Length of neighbors
    
    Return:
        Weighted velocities
    
    '''
    oh = [0] * length
    distance = list(distance)
    for i in x:
        oh[i] = scale - distance.pop(0)
    return np.array(oh)


def generate_grid(xlim=(-1, 1), ylim=(-1, 1), density: int = 20):
    '''
    Generate grid to project velocities.
    
    Args:
        xlim: 
            Grid bound on x axis
        ylim: 
            Grid bound on y axis
        density: 
            How much grid to split
        
    Return:
        grid_X, grid_Y, grid_XY
    
    '''
    Xg, Yg = np.mgrid[
        xlim[0] : xlim[1] : density * 1j, ylim[0] : ylim[1] : density * 1j
    ]
    grid = []
    for i in range(density):
        for j in range(density):
            grid.append([Xg[i][j], Yg[i][j]])
    grid = np.array(grid)
    return Xg, Yg, grid


def velocity_embedding_to_grid(
    pts:'numpy.array',
    vel:'numpy.array',
    nn: "str:knn, radius" = "radius",
    grid_density: int = 20,
    n_neighbors: int = 4,
    radius: float = 2,
    xlim=(None, None),
    ylim=(None, None),
):
    '''
    Project velocities to grid
    
    Args:
        pts: 
            UMAP/tSNE coordinates
        vel: 
            Velocity vector
        nn: 
            knn or radius neighbors to use
        grid_density: 
            density of the grid
        n_neighbors: 
            How much neighbors, works when nn=='knn'
        radius: 
            How large radius, works when nn='radius'
        xlim: 
            Grid bound on x axis
        ylim: 
            Grid bound on y axis
    
    Return:
        
    '''
    x = pts

    if xlim[0]:
        rmv = []
        for ind, i in enumerate(x):
            if i[0] < xlim[0] or i[0] > xlim[1]:
                rmv.append(ind)
        x = np.delete(x, rmv, axis=0)
        vel = np.delete(vel, rmv, axis=0)
    if ylim[0]:
        rmv = []
        for ind, i in enumerate(x):
            if i[1] < ylim[0] or i[1] > ylim[1]:
                rmv.append(ind)
        x = np.delete(x, rmv, axis=0)
        vel = np.delete(vel, rmv, axis=0)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors, radius=radius).fit(x)
    xrange = max(x[:, 0]) - min(x[:, 0])
    yrange = max(x[:, 1]) - min(x[:, 1])
    Xg, Yg, grid = generate_grid(
        xlim=(min(x[:, 0]) - 0.05 * xrange, max(x[:, 0]) + 0.05 * xrange),
        ylim=(min(x[:, 1]) - 0.05 * yrange, max(x[:, 1]) + 0.05 * yrange),
        density=grid_density,
    )
    if nn == "knn":
        distances, indices = nbrs.kneighbors(grid)
    else:
        distances, indices = nbrs.radius_neighbors(grid)

    vel_grid = []
    for i, d in zip(indices, distances):
        if len(i) > 3:
            vel_grid.append(get_weight(i, d, radius, vel.shape[0]).dot(vel) / len(i))
        else:
            vel_grid.append(np.zeros(2))
    vel_grid = np.array(vel_grid)
    lengths = np.sqrt((vel_grid**2).sum(0))
    linewidth = 2 * lengths / lengths[~np.isnan(lengths)].max()
    Ug = vel_grid[:, 0].reshape(grid_density, grid_density)
    Vg = vel_grid[:, 1].reshape(grid_density, grid_density)
    return Xg, Yg, Ug, Vg


def velocity_plot(
    pts,
    vel,
    ax,
    figtype: "str:stream, grid, point" = "grid",
    nn: "str:knn, radius" = "radius",
    grid_density: int = 20,
    n_neighbors: int = 4,
    radius: float = 2,
    streamdensity: float = 1.5,
    xlim=(None, None),
    ylim=(None, None),
    **kwargs
):
    '''
    Project velocities into embedding
    
    Args:
        pts: 
            UMAP/tSNE coordinates
        vel: 
            Velocity vector
        ax: 
            matplotlib.axes
        figtype: 
            'stream', 'grid' or 'point'(single cell)
        nn: 
            knn or radius neighbors to use
        grid_density: 
            density of the grid
        n_neighbors: 
            How much neighbors, works when nn=='knn'
        radius: 
            How large radius, works when nn='radius'
        streamdensity: 
            Density of streamplot, works when figtype==stream
        xlim: 
            Grid bound on x axis
        ylim: 
            Grid bound on y axis
        
    Return:
        matplotlib.axes
    
    '''
    x = pts

    headwidth = kwargs.pop("headwidth", 3)
    headlength = kwargs.pop("headlength", 5)

    if figtype == "point":
        ax.quiver(
            x[:, 0],
            x[:, 1],
            vel[:, 0],
            vel[:, 1],
            headwidth=headwidth,
            headlength=headlength,
        )
        return ax
    Xg, Yg, Ug, Vg = velocity_embedding_to_grid(
        pts, vel, nn, grid_density, n_neighbors, radius, xlim, ylim
    )

    if figtype == "grid":
        ax.quiver(Xg, Yg, Ug, Vg, zorder=3, headwidth=headwidth, headlength=headlength)

    if figtype == "stream":

        lw_coef = kwargs.pop("lw_coef", 1)
        linewidth = kwargs.pop("linewidth", lw_coef * np.sqrt(Ug.T**2 + Vg.T**2))
        arrowsize = kwargs.pop("arrowsize", 1)
        minlength = kwargs.pop("minlength", 0.1)
        maxlength = kwargs.pop("maxlength", 4)
        color = kwargs.pop("color", "k")
        cmap = kwargs.pop("cmap", None)

        ax.streamplot(
            Xg.T,
            Yg.T,
            Ug.T,
            Vg.T,
            linewidth=linewidth,
            density=streamdensity,
            color=color,
            zorder=3,
            arrowsize=arrowsize,
            minlength=minlength,
            maxlength=maxlength,
            cmap=cmap,
        )
    return ax


def plot_tree(
    tree,
    colors,
    ax:'matplotlib.axes',
    colortab:list=["gray", "blue", "green", "orange", "purple"],
    stain:"str: 'all' or 'terminals'"="all",
):
    """
    Draw phylogenetic tree
    
    Args:
        tree: 
            Load from loadtree
        colors:  
            Load from loadtree
        ax: 
            matplotlib axes to draw on
        colortab: 
            A list of colors to paint different cell types
        stain: 
            'all' for color all branches, 'terminals' for color only terminals branches

    Return:
        matplotlib.axes
    """
    colortab = colortab
    names = [i[:-2] for i in colors]
    states = [int(i[-1]) for i in colors]
    states = dict(zip(names, states))
    for i in colors:
        tree.common_ancestor({"name": i[:-2]}).color = colortab[int(i[-1])]

    def cid_ext(x):
        x = str(x.name)
        return

    if stain == "terminals":
        for c in tree.get_nonterminals():
            c.color = "#cdcdcd"
    Phylo.draw(tree, label_func=cid_ext, do_show=False, axes=ax)
    return ax


def mullerplot(data:'numpy.ndarray', label:list, color:list, absolute:bool=0, alpha:float=0.8, ax:'matplotlib.axes'=None):
    '''
    Draw mullerplot
    
    Args:
        data: 
            Population size array. rows for cell type, columns for time point
        latel: 
            Cell type names
        color: 
            Colors list
        absolute: 
            False: show frequency; True: show cell number
        alpha: 
            [0-1], transparent 
        ax: 
            axes to draw mullerplot
        
    Return:
        matplotlib.axes
    '''
    if absolute:
        norm = max(np.sum(data, axis=0))
    else:
        norm = np.sum(data, axis=0)
    data_normed = data / norm
    mp = np.cumsum(data_normed, axis=0)
    adjustpos = 0.5 * (1 - mp[-1])

    # plt.plot(mp.T,color='black', lw=1)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    if absolute:
        ax.fill_between(
            range(data.shape[1]),
            np.zeros(data.shape[1]) + adjustpos,
            mp[0] + adjustpos,
            label=label[0],
            alpha=alpha,
            color=color[0],
        )
        for i in range(mp.shape[0] - 1):
            ax.fill_between(
                range(data.shape[1]),
                mp[i] + adjustpos,
                mp[i + 1] + adjustpos,
                label=label[i + 1],
                alpha=alpha,
                color=color[i + 1],
            )
    else:
        ax.fill_between(
            range(data.shape[1]), mp[0], label=label[0], alpha=alpha, color=color[0]
        )
        for i in range(mp.shape[0] - 1):
            ax.fill_between(
                range(data.shape[1]),
                mp[i],
                mp[i + 1],
                label=label[i + 1],
                alpha=alpha,
                color=color[i + 1],
            )
    # ax.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad = 0)
    return ax

