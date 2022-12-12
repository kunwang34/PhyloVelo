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


class VelocityEmbedding:

    def __init__(self, count, xdr, v):
        self.count = count
        self.xdr = xdr
        self.kNN = None
        self.v = -np.array([v])
        self.d = self.rho(self.v)
        self.neighs_log = {}

    def rho(self, x):
        return np.sign(x) * np.sqrt(abs(x))

    def get_neighbors(self, kNN):
        self.kNN = kNN
        neigh = NearestNeighbors(n_neighbors=kNN)
        neigh.fit(self.xdr)
        self.neigh = neigh

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


def velocity_embedding(sd:'scData', target:str="count", n_neigh:int=None):
    '''
    Project velocity into embedding
    
    Args:
        sd:
            scData
        target: 
            count or x_normed
        n_neigh: 
            kNN pooling. Default: Ncells//3
    '''
    data = getattr(sd, target)
    ve = VelocityEmbedding(data.to_numpy(), sd.Xdr.to_numpy(), sd.velocity)
    if n_neigh is None:
        n_neigh = data.shape[0] // 3
    ve.get_neighbors(n_neigh)
    ve.transit_mat()
    v = []
    for i in range(data.shape[0]):
        v.append(ve.project(i))
    v = np.array(v)
    v_norm = np.linalg.norm(v, axis=1)
    cf = np.quantile(v_norm, 0.8)
    v[v_norm > cf] = ((cf / v_norm[v_norm > cf]) * v[v_norm > cf].T).T
    sd.velocity_embeded = v
    return sd
