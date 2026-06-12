import warnings
from multiprocessing import Pool

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.autonotebook import tqdm


def paired_correlation_rows(A: np.array, B: np.array) -> np.array:
    '''
    Calculate paired correlation
    
    Args:
        A: numpy.array
        B: numpy.array
    
    Return:
        numpy.array
    '''
    A_m = A - A.mean(1)[:, None]
    B_m = B - B.mean(1)[:, None]
    return (A_m * B_m).sum(1) / (np.linalg.norm(A_m, 2, 1) * np.linalg.norm(B_m, 2, 1))


def _as_float_array(x):
    return np.asarray(x, dtype=np.float64)


class VelocityEmbedding:

    def __init__(self, count, xdr, v):
        self.count = _as_float_array(count)
        self.xdr = _as_float_array(xdr)
        self.kNN = None
        self.v = -_as_float_array(v)[None, :]
        self.d = self.rho(self.v)
        d = self.d.ravel()
        self._d_mean = d.mean()
        self._d_centered = d - self._d_mean
        self._d_norm = np.linalg.norm(self._d_centered)
        self._d_nonzero = np.flatnonzero(d)
        self._d_nonzero_values = d[self._d_nonzero]
        self._neighbor_indices = None
        self.neighs_log = {}

    def rho(self, x):
        return np.sign(x) * np.sqrt(abs(x))

    def get_neighbors(self, kNN):
        self.kNN = kNN
        neigh = NearestNeighbors(n_neighbors=kNN)
        neigh.fit(self.xdr)
        self.neigh = neigh
        self._neighbor_indices = None

    def _get_neighbor_indices(self):
        if self._neighbor_indices is not None:
            return self._neighbor_indices

        neighs = self.neigh.kneighbors(self.xdr, return_distance=False)
        neighs.sort(axis=1)
        keep = neighs != np.arange(neighs.shape[0])[:, None]
        n_neighs = keep.sum(axis=1)
        if np.all(n_neighs == n_neighs[0]):
            neighs = neighs[keep].reshape(neighs.shape[0], n_neighs[0])
            self._neighbor_indices = neighs
        else:
            self._neighbor_indices = [row[mask] for row, mask in zip(neighs, keep)]

        return self._neighbor_indices

    def _auto_chunk_size(self, n_neighs, n_genes, chunk_size=None):
        if chunk_size is not None:
            return max(1, int(chunk_size))

        bytes_per_cell = max(1, n_neighs * n_genes * 8)
        # Count gather and transformed working arrays can coexist.
        return max(1, int((64 * 1024**2) / (3 * bytes_per_cell)))

    def _transition_probabilities(self, diff_vecs):
        abs_diff = np.abs(diff_vecs)
        diff_vecs = np.sign(diff_vecs) * np.sqrt(abs_diff)
        diff_mean = diff_vecs.mean(axis=-1)
        diff_sum = diff_mean * diff_vecs.shape[-1]
        diff_norm_sq = abs_diff.sum(axis=-1) - diff_vecs.shape[-1] * diff_mean**2
        diff_norm_sq = np.maximum(diff_norm_sq, 0)

        if len(self._d_nonzero) == diff_vecs.shape[-1]:
            numerator = np.tensordot(diff_vecs, self._d_centered, axes=([-1], [0]))
        else:
            numerator = np.tensordot(
                diff_vecs[..., self._d_nonzero], self._d_nonzero_values, axes=([-1], [0])
            )
            numerator -= self._d_mean * diff_sum

        numerator = np.where(diff_norm_sq == 0, 0, numerator)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = numerator / (np.sqrt(diff_norm_sq) * self._d_norm)
        return np.exp(corr / 10)

    def _project_neighbor_block(self, start, stop, neighs):
        diff_vecs = self.count[neighs] - self.count[start:stop, None, :]
        probabilities = self._transition_probabilities(diff_vecs)
        weights = probabilities / probabilities.sum(axis=1, keepdims=True)

        diff_emb = self.xdr[neighs] - self.xdr[start:stop, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            diff_emb = diff_emb / np.sqrt((diff_emb**2).sum(axis=2, keepdims=True))

        n_neighs = neighs.shape[1]
        return ((weights - 1 / n_neighs)[:, :, None] * diff_emb).sum(axis=1), probabilities

    def project_all(self, chunk_size=None, store_transition=False):
        if not self.kNN:
            self.transit_mat()
            return np.array([self.project(i) for i in range(self.count.shape[0])])

        neighs = self._get_neighbor_indices()
        if not isinstance(neighs, np.ndarray):
            self.transit_mat()
            return np.array([self.project(i) for i in range(self.count.shape[0])])

        n_cells, n_neighs = neighs.shape
        if n_neighs == 0:
            return np.zeros((n_cells, self.xdr.shape[1]), dtype=np.float64)

        chunk_size = self._auto_chunk_size(n_neighs, self.count.shape[1], chunk_size)
        projected = np.empty((n_cells, self.xdr.shape[1]), dtype=np.float64)
        transitions = [] if store_transition else None

        for start in tqdm(range(0, n_cells, chunk_size)):
            stop = min(start + chunk_size, n_cells)
            block, probabilities = self._project_neighbor_block(start, stop, neighs[start:stop])
            projected[start:stop] = block
            if store_transition:
                transitions.extend(probabilities)

        if store_transition:
            self.p = transitions
            self.neighs_log = {i: neighs[i] for i in range(n_cells)}

        return projected

    def transit_mat1(self, i):
        if self.kNN:
            neighs = self.neigh.kneighbors([self.xdr[i]], return_distance=False)[0]
            neighs.sort()
            neighs = neighs[neighs != i]
            self.neighs_log[i] = neighs
            diff_vecs = self.count[neighs] - self.count[i]
        else:
            diff_vecs = self.count - self.count[i]
        return np.exp(paired_correlation_rows(self.rho(diff_vecs), self.d) / 10)

    def transit_mat(self, n_process=0):
        n = self.count.shape[0]
        if self.kNN and not n_process:
            neighs = self._get_neighbor_indices()
            if isinstance(neighs, np.ndarray):
                n_cells, n_neighs = neighs.shape
                chunk_size = self._auto_chunk_size(n_neighs, self.count.shape[1])
                mat = []
                for start in tqdm(range(0, n_cells, chunk_size)):
                    stop = min(start + chunk_size, n_cells)
                    diff_vecs = self.count[neighs[start:stop]] - self.count[start:stop, None, :]
                    mat.extend(self._transition_probabilities(diff_vecs))
                self.p = mat
                self.neighs_log = {i: neighs[i] for i in range(n_cells)}
                return

        if n_process:
            with Pool(n_process) as p:
                mat = list(tqdm(p.imap(self.transit_mat1, range(n)), total=n))
        else:
            mat = []
            for i in tqdm(range(n)):
                mat.append(self.transit_mat1(i))
        self.p = mat

    def project(self, i):
        dx = np.zeros(2)
        if self.kNN:
            weight = self.p[i]
            weight = weight / weight.sum()
            n = len(weight)
            for cnt, j in enumerate(self.neighs_log[i]):
                wi = weight[cnt]
                cnt += 1
                diff = self.xdr[j] - self.xdr[i]
                dx = dx + (wi - 1 / n) * diff / np.sqrt((diff**2).sum())

        else:
            weight = np.delete(self.p[i], i)
            weight = weight / weight.sum()
            weight = np.insert(weight, i, 0)
            n = len(weight)
            for j in range(n):
                if j != i:
                    diff = self.xdr[j] - self.xdr[i]
                    dx = dx + (weight[j] - 1 / n) * diff / np.sqrt((diff**2).sum())
        return dx


def velocity_embedding(sd:'scData', target:str="count", n_neigh:int=None, chunk_size:int=None):
    '''
    Project velocity into embedding
    
    Args:
        sd:
            scData
        target: 
            count or x_normed
        n_neigh: 
            kNN pooling. Default: Ncells//3
        chunk_size:
            Number of cells per vectorized block. Default estimates a memory-safe size.
    '''
    data = getattr(sd, target)
    ve = VelocityEmbedding(data.to_numpy(), sd.Xdr.to_numpy(), sd.velocity)
    if n_neigh is None:
        n_neigh = data.shape[0] // 3
    ve.get_neighbors(n_neigh)
    v = ve.project_all(chunk_size=chunk_size)
    v_norm = np.linalg.norm(v, axis=1)
    cf = np.quantile(v_norm, 0.8)
    v[v_norm > cf] = ((cf / v_norm[v_norm > cf]) * v[v_norm > cf].T).T
    sd.velocity_embeded = v
    return sd
