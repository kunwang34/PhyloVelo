import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm


def get_nearest_neighbor(data:'numpy.ndarray', target:int, n_neighbors:int=10):
    '''
    Get nearest neighbors of the target
    
    Args:
        data:
            Data to train knn
        target:
            Target point to get nearest neighbors
        n_neighbors:
            How many nearest neighbors to return
    
    Returns:
        list:
            Euclidean distance from target to neighbors
        list:
            Neighbors' indices
    
    '''
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(data)
    neighbor = neigh.kneighbors([data[target]])
    distance, loc = neighbor[0], neighbor[1]
    return distance[0], loc[0]


def time_interval(pt1:'numpy.ndarry', pt2:'numpy.ndarry', v1:'numpy.ndarry', v2:'numpy.ndarry'):
    '''
    Given two points' coordinate and velocity, calculate the time interval
    
    Args:
        pt1: 
            Coordinate of one point
        pt2: 
            Coordinate of the other point
        v1: 
            Velocity of one point
        v2: 
            Velocity of the other point
    
    Return:
        float:
            Time interval
    '''
    va = (v1 + v2) / 2
    a1 = pt2 - pt1
    with np.errstate(all="ignore"):
        cos = (a1 * va).sum() / np.sqrt((a1**2).sum()) / np.sqrt((va**2).sum())
    v_proj = np.sqrt((va**2).sum()) * cos
    pt = np.sqrt((a1**2).sum()) / (v_proj + 1e-9)
    if np.isnan(pt):
        return 1e99
    return pt


def graph_dict(pts:'numpy.ndarry', v:'numpy.ndarry', n_neighbors:int=30):
    '''
    Build graph to construct MST
    
    Args:
        pts: 
            All cells' coordinate in embedding
        v:
            Phylo velocity
        n_neighbors:
            N nearest neighbors to build MST
    
    Return:
        dict:
            Graph to build MSt
    '''
    graph_dict = {}
    for i in range(pts.shape[0]):
        graph_dict[i] = {}
        dis, nbrs = get_nearest_neighbor(pts, i, n_neighbors)
        for d, nbr in zip(dis, nbrs):
            graph_dict[i][nbr] = abs(time_interval(pts[i], pts[nbr], v[i], v[nbr]))
    return graph_dict


def prim(graph, root):
    '''
    Prim algorithm to build MST from graph
    '''
    assert type(graph) == dict
    nodes = set(graph.keys())
    nodes.remove(root)
    visited = [root]
    path = []
    next = None
    with tqdm(total=len(nodes)) as pbar:
        while nodes:
            distance = float("inf")
            for s in visited:
                for d in graph[s]:
                    if d in visited or s == d:
                        continue
                    if graph[s][d] < distance:
                        distance = graph[s][d]
                        pre = s
                        next = d
            path.append((pre, next))
            visited.append(next)
            nodes.remove(next)
            pbar.update(1)
    return path


def calc_phylo_pseudotime(sd:'scData', n_neighbors:int=30, r_sample:float=1):
    '''
    Calculate the phyloVelo pseudotime
    
    Args:
        sd:
            sc data
        n_neighbors:
            N nearest neighbors to build MST. The smaller the number, the faster the calculation, but there is a chance of error
        r_sample:
            [0-1], random sample a subset calculate pseudotime.
            
    Return:
        scData.phylo_pseudotime
    '''
    if r_sample < 1:
        n_cells = len(sd.Xdr.index)
        n_sample = int(n_cells * r_sample)
        sample_names = np.random.choice(sd.Xdr.index, n_sample, replace=False)
        pts = sd.Xdr.loc[sample_names].to_numpy()
        v = sd.velocity_embeded[np.isin(sd.Xdr.index, sample_names), :]
    else:
        sample_names = sd.Xdr.index
        pts = sd.Xdr.to_numpy()
        v = sd.velocity_embeded

    with np.errstate(all="ignore"):
        graph = graph_dict(pts, v, n_neighbors=n_neighbors)
    path = prim(graph, 0)
    pseudo_time = dict()
    for i in range(pts.shape[0]):
        pseudo_time[sample_names[i]] = 1e99
        pseudo_time[sample_names[0]] = 1
    for i in path:
        pseudo_time[sample_names[i[1]]] = min(
            pseudo_time[sample_names[i[0]]]
            + time_interval(pts[i[0]], pts[i[1]], v[i[0]], v[i[1]]),
            pseudo_time[sample_names[i[1]]],
        )

    if len(path) < sd.Xdr.shape[0]:
        pseudo_time_full = dict()
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(pts)

        for ind, name in enumerate(sd.Xdr.index):
            if name in pseudo_time:
                pseudo_time_full[name] = pseudo_time[name]
            else:
                # dis, neigh = get_nearest_neighbor(xdr, sd.Xdr.loc[name], n_neighbors=5)
                neighbor = neigh.kneighbors([sd.Xdr.loc[name].to_numpy()])
                neighbors = neighbor[1][0]
                pseudo_time_full[name] = np.mean(
                    [pseudo_time[sample_names[i]] for i in neighbors]
                )
        for ind, name in enumerate(sd.Xdr.index):
            pseudo_time[ind] = pseudo_time_full[name]

    time = np.array([-pseudo_time[i] for i in range(sd.Xdr.shape[0])])
    time = time - min(time)
    time = time / max(time)
    sd.phylo_pseudotime = time
    return sd
