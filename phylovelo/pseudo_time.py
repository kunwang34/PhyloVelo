from heapq import heappop, heappush

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


def _as_float_array(x):
    return np.asarray(x, dtype=np.float64)


def _normalize_01(x, robust_quantiles=None):
    x = _as_float_array(x).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float64)

    if robust_quantiles is not None:
        lo_q, hi_q = robust_quantiles
        lo, hi = np.nanquantile(x[finite], [lo_q, hi_q])
        x = np.clip(x, lo, hi)

    finite = np.isfinite(x)
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi <= lo:
        out = np.zeros_like(x, dtype=np.float64)
    else:
        out = (x - lo) / (hi - lo)
    out[~np.isfinite(out)] = np.nanmedian(out[np.isfinite(out)]) if np.isfinite(out).any() else 0
    return out


def _normalize_with_bounds(x, lo, hi):
    x = _as_float_array(x).copy()
    if hi <= lo:
        out = np.zeros_like(x, dtype=np.float64)
    else:
        out = (np.clip(x, lo, hi) - lo) / (hi - lo)
    out[~np.isfinite(out)] = np.nanmedian(out[np.isfinite(out)]) if np.isfinite(out).any() else 0
    return out


def _time_intervals(pt1, pt2, v1, v2):
    pt1 = _as_float_array(pt1)
    pt2 = _as_float_array(pt2)
    v1 = _as_float_array(v1)
    v2 = _as_float_array(v2)

    diff = pt2 - pt1
    va = (v1 + v2) / 2
    distance = np.sqrt((diff**2).sum(axis=-1))
    v_proj = (diff * va).sum(axis=-1) / (distance + 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        interval = distance / (v_proj + 1e-9)
    return np.where(np.isfinite(interval), interval, 1e99)


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
    return float(_time_intervals(pt1, pt2, v1, v2))


def _build_knn_adjacency(pts, v, n_neighbors=30):
    n_cells = pts.shape[0]
    adjacency = [[] for _ in range(n_cells)]
    if n_cells <= 1:
        return adjacency

    n_neighbors = min(max(2, int(n_neighbors)), n_cells)
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(pts)
    _, indices = neigh.kneighbors(pts)

    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices.reshape(-1)
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    intervals = _time_intervals(pts[rows], pts[cols], v[rows], v[cols])

    for i, j, interval in zip(rows, cols, intervals):
        weight = abs(interval)
        adjacency[i].append((j, weight, interval))
        adjacency[j].append((i, weight, -interval))

    return adjacency


def _connected_components(adjacency):
    n_cells = len(adjacency)
    visited = np.zeros(n_cells, dtype=bool)
    components = []

    for start in range(n_cells):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor, _, _ in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.array(component, dtype=int))

    return components


def _bridge_components(adjacency, pts, v, root=0):
    components = _connected_components(adjacency)
    if len(components) <= 1:
        return adjacency

    root_component = next(i for i, component in enumerate(components) if root in component)
    ordered = [components[root_component]] + [
        component for i, component in enumerate(components) if i != root_component
    ]
    connected = ordered[0].copy()

    for component in ordered[1:]:
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(pts[connected])
        distances, nearest = neigh.kneighbors(pts[component])
        local = int(np.argmin(distances[:, 0]))
        src = int(connected[nearest[local, 0]])
        dst = int(component[local])
        interval = time_interval(pts[src], pts[dst], v[src], v[dst])
        weight = abs(interval)
        adjacency[src].append((dst, weight, interval))
        adjacency[dst].append((src, weight, -interval))
        connected = np.concatenate((connected, component))

    return adjacency


def _prim_edges(adjacency, root=0):
    n_cells = len(adjacency)
    if n_cells <= 1:
        return []

    visited = np.zeros(n_cells, dtype=bool)
    visited[root] = True
    visited_count = 1
    heap = []
    path = []

    for neighbor, weight, interval in adjacency[root]:
        heappush(heap, (weight, root, neighbor, interval))

    with tqdm(total=n_cells - 1) as pbar:
        while visited_count < n_cells:
            if not heap:
                unvisited = np.flatnonzero(~visited)
                if len(unvisited) == 0:
                    break
                next_root = int(unvisited[0])
                visited[next_root] = True
                visited_count += 1
                pbar.update(1)
                for neighbor, weight, interval in adjacency[next_root]:
                    heappush(heap, (weight, next_root, neighbor, interval))
                continue

            _, src, dst, interval = heappop(heap)
            if visited[dst]:
                continue

            visited[dst] = True
            visited_count += 1
            path.append((src, dst, interval))
            pbar.update(1)

            for neighbor, weight, next_interval in adjacency[dst]:
                if not visited[neighbor]:
                    heappush(heap, (weight, dst, neighbor, next_interval))

    return path


def _graph_pseudotime(pts, v, n_neighbors=30, root=0, robust_quantiles=(0.01, 0.99)):
    if pts.shape[0] == 0:
        return np.array([])
    if pts.shape[0] == 1:
        return np.zeros(1)

    adjacency = _build_knn_adjacency(pts, v, n_neighbors=n_neighbors)
    adjacency = _bridge_components(adjacency, pts, v, root=root)
    path = _prim_edges(adjacency, root=root)

    pseudo_time = np.full(pts.shape[0], np.nan)
    pseudo_time[root] = 0
    for src, dst, interval in path:
        if np.isfinite(pseudo_time[src]):
            pseudo_time[dst] = pseudo_time[src] + interval

    missing = ~np.isfinite(pseudo_time)
    if missing.any():
        pseudo_time[missing] = np.nanmedian(pseudo_time[~missing]) if (~missing).any() else 0

    return _normalize_01(-pseudo_time, robust_quantiles=robust_quantiles)


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
    pts = _as_float_array(pts)
    v = _as_float_array(v)
    adjacency = _build_knn_adjacency(pts, v, n_neighbors=n_neighbors)
    return {
        i: {neighbor: weight for neighbor, weight, _ in neighbors}
        for i, neighbors in enumerate(adjacency)
    }


def prim(graph, root):
    '''
    Prim algorithm to build MST from graph
    '''
    assert type(graph) == dict
    adjacency = [[] for _ in range(len(graph))]
    for src, neighbors in graph.items():
        for dst, weight in neighbors.items():
            adjacency[src].append((dst, weight, weight))
    return [(src, dst) for src, dst, _ in _prim_edges(adjacency, root)]


def _sample_cells(index, r_sample, random_state=None):
    n_cells = len(index)
    if r_sample >= 1:
        return np.asarray(index)

    n_sample = max(1, int(n_cells * r_sample))
    rng = np.random.default_rng(random_state)
    return rng.choice(np.asarray(index), n_sample, replace=False)


def _positions_for_names(index, names):
    index = np.asarray(index)
    name_to_pos = {name: i for i, name in enumerate(index)}
    return np.array([name_to_pos[name] for name in names], dtype=int)


def _get_expression_data(obj, target):
    if obj is None:
        return None
    if hasattr(obj, "columns") and hasattr(obj, "iloc"):
        return obj
    return getattr(obj, target)


def _meg_pseudotime_scores(X, lo, spread, signs, weights, aggregation):
    gene_time = np.clip((X - lo) / spread, 0, 1)
    gene_time[:, signs < 0] = 1 - gene_time[:, signs < 0]

    if aggregation == "weighted_mean":
        if not np.isfinite(weights).any() or weights.sum() <= 0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()
        return np.nansum(gene_time * weights, axis=1)
    if aggregation == "median":
        return np.nanmedian(gene_time, axis=1)
    raise ValueError("aggregation must be 'median' or 'weighted_mean'.")


def calc_phylo_pseudotime(
    sd:'scData',
    n_neighbors:int=30,
    r_sample:float=1,
    method:str="graph",
    target:str="x_normed",
    random_state:int=None,
):
    '''
    Calculate the phyloVelo pseudotime
    
    Args:
        sd:
            sc data
        n_neighbors:
            N nearest neighbors to build MST. The smaller the number, the faster the calculation, but there is a chance of error
        r_sample:
            [0-1], random sample a subset calculate pseudotime.
        method:
            'graph' uses embedding velocities and a kNN MST; 'meg' uses robust MEG expression.
        target:
            Expression matrix used when method='meg'.
        random_state:
            Seed for subsampling.
            
    Return:
        scData.phylo_pseudotime
    '''
    method = method.lower()
    if method in ["meg", "expression", "expr"]:
        return calc_meg_pseudotime(sd, target=target)
    if method != "graph":
        raise ValueError("method must be 'graph' or 'meg'.")

    sample_names = _sample_cells(sd.Xdr.index, r_sample, random_state=random_state)
    if len(sample_names) < len(sd.Xdr.index):
        sample_pos = _positions_for_names(sd.Xdr.index, sample_names)
        pts = sd.Xdr.loc[sample_names].to_numpy()
        v = sd.velocity_embeded[sample_pos, :]
    else:
        sample_names = np.asarray(sd.Xdr.index)
        pts = sd.Xdr.to_numpy()
        v = sd.velocity_embeded

    pts = _as_float_array(pts)
    v = _as_float_array(v)
    sample_time = _graph_pseudotime(pts, v, n_neighbors=n_neighbors)

    if len(sample_names) < len(sd.Xdr.index):
        n_fill_neighbors = min(5, len(sample_names))
        neigh = NearestNeighbors(n_neighbors=n_fill_neighbors)
        neigh.fit(pts)
        distances, neighbors = neigh.kneighbors(sd.Xdr.to_numpy())
        weights = 1 / (distances + 1e-9)
        time = (weights * sample_time[neighbors]).sum(axis=1) / weights.sum(axis=1)
        time[sample_pos] = sample_time
    else:
        time = sample_time

    sd.phylo_pseudotime = _normalize_01(time, robust_quantiles=(0.01, 0.99))
    return sd


def calc_meg_pseudotime(
    sd:'scData',
    target:str="x_normed",
    genes:list=None,
    robust_quantiles:tuple=(0.05, 0.95),
    aggregation:str="median",
    min_genes:int=3,
    query_data=None,
    query_sd:'scData'=None,
    query_target:str=None,
):
    '''
    Calculate pseudotime directly from robustly oriented MEG expression.

    Args:
        sd:
            sc data
        target:
            Expression matrix to use, usually 'x_normed' or 'count'.
        genes:
            MEGs to use. Default uses sd.megs.
        robust_quantiles:
            Lower and upper quantiles used to clip per-gene expression.
        aggregation:
            'median' for robust L1 aggregation or 'weighted_mean'.
        min_genes:
            Minimum number of usable MEGs.
        query_data:
            Independent expression matrix (cells x genes) to score with the
            reference dataset's MEGs, velocity directions, and robust scaling.
        query_sd:
            Independent scData object. Uses query_target or target as expression
            matrix and writes query pseudotime to query_sd.phylo_pseudotime.
        query_target:
            Expression matrix name for query_sd. Default: same as target.

    Return:
        sd if no query is provided; query pseudotime array if query_data is
        provided; query_sd if query_sd is provided.
    '''
    data = getattr(sd, target)
    query_target = target if query_target is None else query_target
    query_expr = _get_expression_data(query_sd, query_target)
    if query_data is not None and query_expr is not None:
        raise ValueError("Pass only one of query_data or query_sd.")
    if query_data is not None:
        query_expr = query_data

    if genes is None:
        genes = getattr(sd, "megs", None)
    if genes is None or len(genes) == 0:
        raise ValueError("No MEGs found. Run velocity_inference first or pass genes.")

    columns = np.asarray(data.columns)
    gene_to_col = {gene: i for i, gene in enumerate(columns)}
    if query_expr is not None:
        query_columns = np.asarray(query_expr.columns)
        query_gene_to_col = {gene: i for i, gene in enumerate(query_columns)}
        genes = [gene for gene in genes if gene in gene_to_col and gene in query_gene_to_col]
    else:
        query_gene_to_col = None
        genes = [gene for gene in genes if gene in gene_to_col]

    if len(genes) < min_genes:
        if query_expr is None:
            raise ValueError("Not enough MEGs are available in the reference expression matrix.")
        raise ValueError(
            "Not enough MEGs are shared by the reference and query expression matrices."
        )

    gene_idx = np.array([gene_to_col[gene] for gene in genes], dtype=int)
    X = _as_float_array(data.iloc[:, gene_idx].to_numpy())

    velocity = _as_float_array(getattr(sd, "velocity", np.ones(len(columns))))
    signs = np.sign(velocity[gene_idx])
    usable = signs != 0
    if usable.sum() < min_genes:
        raise ValueError("Not enough selected MEGs have non-zero velocity direction.")

    X = X[:, usable]
    signs = signs[usable]
    weights = np.abs(velocity[gene_idx][usable])
    genes = list(np.asarray(genes)[usable])

    lo, hi = np.nanquantile(X, robust_quantiles, axis=0)
    spread = hi - lo
    usable = np.isfinite(spread) & (spread > 1e-12)
    if usable.sum() < min_genes:
        raise ValueError("Not enough selected MEGs have usable expression variation.")

    X = X[:, usable]
    signs = signs[usable]
    weights = weights[usable]
    lo, hi, spread = lo[usable], hi[usable], spread[usable]
    genes = list(np.asarray(genes)[usable])

    time_score = _meg_pseudotime_scores(X, lo, spread, signs, weights, aggregation)
    finite = np.isfinite(time_score)
    if finite.any():
        score_lo, score_hi = np.nanquantile(time_score[finite], robust_quantiles)
    else:
        score_lo, score_hi = 0, 1
    sd.phylo_pseudotime = _normalize_with_bounds(time_score, score_lo, score_hi)

    if query_expr is None:
        return sd

    query_gene_idx = np.array([query_gene_to_col[gene] for gene in genes], dtype=int)
    X_query = _as_float_array(query_expr.iloc[:, query_gene_idx].to_numpy())
    query_score = _meg_pseudotime_scores(X_query, lo, spread, signs, weights, aggregation)
    query_time = _normalize_with_bounds(query_score, score_lo, score_hi)

    if query_sd is not None:
        query_sd.phylo_pseudotime = query_time
        return query_sd

    return query_time
