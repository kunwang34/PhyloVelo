import cmath

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial, gamma
from scipy.stats import chi2, mannwhitneyu, nbinom, norm, spearmanr, pearsonr
from tqdm.autonotebook import tqdm
from statsmodels.stats.multitest import fdrcorrection

def mle_zinb(data:list):
    '''
    Maximum likelihood estimation of ZINB distribution
    
    Args:
        data:
            Data
    Returns:
        tuple:
            (-loglikelihood, MLE)
    '''
    data = np.array(data)

    def lh(theta, data):

        psi, mu, n = theta
        p = n / (mu + n + 1e-9)
        n_zeros = np.sum(data == 0)
        non_zeros = data[data != 0]
        np.seterr(all="ignore")
        pmf0 = -n_zeros * np.log((1 - psi) + psi * (n / (n + mu)) ** n)
        pmf1 = -sum(nbinom(n, p).logpmf(non_zeros)) - len(non_zeros) * np.log(
            psi + 1e-9
        )
        return pmf0 + pmf1

    init_val = (0.3, np.mean(data), 1)
    res = minimize(lh, init_val, data, bounds=((0, 1), (0, np.inf), (0, np.inf)))
    return (-lh(res.x, data), res.x)


def mle_nb(data):
    '''
    Maximum likelihood estimation of NB distribution
    
    Args:
        data:
            Data
    Returns:
        tuple:
            (-loglikelihood, MLE)
    '''
    data = np.array(data)

    def lh(theta, data):
        mu, n = theta
        p = n / (mu + n + 1e-9)
        return -sum(nbinom(n, p).logpmf(data))

    init_val = (np.mean(data), 1)
    res = minimize(lh, init_val, data, bounds=((0, np.inf), (0, np.inf)))
    return (-lh(res.x, data), res.x)


def mle_norm(data):
    '''
    Maximum likelihood estimation of Normal distribution
    
    Args:
        data:
            Data
    Returns:
        tuple:
            (-loglikelihood, MLE)
    '''
    data = np.array(data)

    def lh(theta, data):
        mu, sigma2 = data
        return sum(norm(mu, np.sqrt(sigma2)).logpdf(data))

    theta = [np.mean(data), np.var(data)]
    return (lh(theta, data), theta)


def mle_zinorm(data):
    '''
    Maximum likelihood estimation of ZI-Normal distribution
    
    Args:
        data:
            Data
    Returns:
        tuple:
            (-loglikelihood, MLE)
    '''
    data = np.array(data)

    def lh(theta, data):
        np.seterr(all="ignore")
        psi, mu, sigma2 = theta
        n_zeros = np.sum(data == 0)
        non_zeros = data[data != 0]
        psi, mu, sigma2 = theta
        np.seterr(all="ignore")
        pmf0 = -n_zeros * np.log(
            (1 - psi) + psi * (np.exp(mu**2) / (np.sqrt(2 * np.pi * sigma2)))
        )
        pmf1 = -sum(norm(mu, np.sqrt(sigma2)).logpdf(non_zeros)) - len(
            non_zeros
        ) * np.log(psi + 1e-9)
        return pmf0 + pmf1

    init_val = (0.5, np.mean(data[data != 0]), np.var(data))
    res = minimize(
        lh, init_val, data, bounds=((0, 1), (-np.inf, np.inf), (-np.inf, np.inf))
    )
    return (-lh(res.x, data), res.x)


def lrtest(lh1:float, lh0:float, n:int, alpha:float):
    '''
    Likelihood ratio test
    
    Args:
        lh1: 
            Likelihood of alternative hypothesis
        lh0:
            Likelihood of null hypothesis
        n:
            Degrees of freedom
        alpha:
            Significance level
    
    Returns:
        tuple:
            (Likelihood ratio statistics, is reject)
    '''
    lr = 2 * (lh1 - lh0)
    return lr, lr > chi2(n).ppf(1 - alpha)


def est_z(x:'ndarray', alpha:float, mu0:float, sigma2:float, model:str):
    '''
    Estimation of latent expression z
    '''
    a, u, s = alpha, mu0, sigma2
    if model == "nb":
        sqrt = (
            4 * (-(a**2) + 3 * a * s - a * u - u**2) ** 3
            + (
                -2 * a**3
                + 9 * a**2 * s
                - 3 * a**2 * u
                - 9 * a * s * u
                + 27 * a * s * x
                + 3 * a * u**2
                + 2 * u**3
            )
            ** 2
        )
        tmp = (
            -2 * a**3
            + 9 * a**2 * s
            - 3 * a**2 * u
            + cmath.sqrt(sqrt)
            - 9 * a * s * u
            + 27 * a * s * x
            + 3 * a * u**2
            + 2 * u**3
        ) ** (1 / 3)
        root1 = (
            0.264567 * tmp
            - (0.419974 * (-(a**2) + 3 * a * s - a * u - u**2)) / tmp
            + 0.333333 * (u - a)
        )
        root2 = (
            -(0.13228 - 0.22912j) * tmp
            + ((0.20999 + 0.36371j) * (-(a**2) + 3 * a * s - a * u - u**2)) / tmp
            + 0.333333 * (u - a)
        )
        root3 = (
            -(0.13228 + 0.22912j) * tmp
            + ((0.20999 - 0.36371j) * (-(a**2) + 3 * a * s - a * u - u**2)) / tmp
            + 0.333333 * (u - a)
        )
        roots = np.array([root1, root2, root3])
        cplxs = np.array([abs(i.imag) for i in roots])
        return float(roots[cplxs == min(cplxs)].real)
    else:
        return (s * x + a * u) / (s + a)


class InfMu:
    '''
    Inference mean of z at time t
    '''
    def __init__(self, time, paras):
        tp = list(set(time))
        tp.sort()
        if len(paras[tp[0]][1]) == 2:
            mus = [paras[i][1][0] for i in tp]

        else:
            mus = [paras[i][1][1] for i in tp]

        x, y = np.array(tp), np.array(mus)
        x_avg, y_avg = np.mean(x), np.mean(y)
        s2 = sum(x**2) / len(x)
        rho = sum(x * y) / len(x)
        self.slope = (rho - x_avg * y_avg) / (s2 - x_avg**2)
        self.inte = y_avg - x_avg * self.slope
        # y_cent = y - (self.slope * x + self.inte)
        self.sigma2 = np.var(y)

    def get_mu(self, t):
        return self.slope * t + self.inte


def latenct_z_inference(data:list, time:int, model:str):
    '''
    Inference latent expression z at time t in given model
    '''
    data = np.array(data)
    tp = list(set(time))
    tp.sort()
    time = np.array(time)
    lh_nb, paras = mle_nb(data)
    lh_zinb, paras = mle_zinb(data)
    paras = dict()
    if lrtest(lh_zinb, lh_nb, 3, 0.05):
        for t in tp:
            paras[t] = mle_zinb(data[time == t])

    else:
        for t in tp:
            paras[t] = mle_nb(data[time == t])
    z_infs = []
    im = InfMu(time, paras)
    for x, t in zip(data, time):
        alpha = paras[t][1][-1]
        z_infs.append(est_z(x, alpha, im.get_mu(t), im.sigma2, model))
    return np.array(z_infs)


def is_meg(x, y, trend=0):
    '''
    Determine whether a gene is meg
    
    Args:
        x: 
            Gene expression
        y:
            Cell generation
        trend:
            positive for incresing megs, negative for decreasing megs
    
    Return:
        bool:
            If a gene is MEG
    '''
    x = np.array(x).astype(int)
    y = np.array(y)
    cond = "greater" if trend > 0 else "less"
    for k in range(min(x), max(x)):
        try:
            if mannwhitneyu(y[x == k], y[x == k + 1], alternative=cond)[1] < 0.05:
                return False
        except:
            None
    return True


def velocity_inference(
    sd:'scData', time:list=None, cutoff:float=0.95, alpha:float=0.05, target:str="x_normed", exact:bool=False
):
    '''
    Inference phylogenetic velocity
    
    Args:
        sd:
            scData
        time:
            if None, cell generation will be automatically calculated from phylo tree 
        cutoff:
            Only calculate genes with top 'cutoff' correlation
        alpha:
            Significance level
        target:
            which data to inference, 'count' for nb model or 'x_normed' for normal model
        exact:
            True to use 'is_meg' function; False do not use
    
    Return:
        sd.velocity
    '''
    assert target in ["x_normed", "count"]

    data = getattr(sd, target)
    coefs = []
    pvals = []
    if target == "x_normed":
        model = "norm"
    else:
        model = "nb"
        
    if time is None:
        tree = sd.phylo_tree
        depths = tree.depths()
        terminals = tree.get_terminals()
        time = np.round([depths[i] for i in terminals])
        
        
    for gene in data.columns:
        coef, pv = spearmanr(data[gene], time)
        coefs.append(coef)
        pvals.append(pv)
    coefs = np.array(coefs)
    pvals = np.array(pvals)

    cutoff = np.quantile(np.abs(coefs), cutoff)
    meg_candidates = []
    velos = dict()
    pearsonr_pvals = []
    
    zs_lat = pd.DataFrame()
    with tqdm(total=data.shape[1]) as pbar:
        for ind, gene in enumerate(data.columns):
            pbar.update(1)
            if exact:
                x = np.array(time)
                y = np.array(data[gene])
                if not is_meg(x, y, trend=coefs[ind]):
                    continue
            if abs(coefs[ind]) > cutoff and pvals[ind] < alpha:
                meg_candidates.append(gene)
                x = np.array(time)
                y = data[gene]
                z = np.array(latenct_z_inference(y, time, model))
                z_lat = pd.DataFrame(data=y, index=data.index, columns=[gene])
                if not zs_lat.shape[0]:
                    zs_lat = z_lat
                else:
                    zs_lat = pd.concat((zs_lat, z_lat), join="inner", axis=1)
                if target == "count":
                    np.seterr(all="ignore")
                    y = np.log(y + 1)
                filterna = ~np.isnan(y)
                x, y = x[filterna], y[filterna]
                pearsonr_pvals.append(pearsonr(x, y)[1])
                x_avg, z_avg = np.mean(x), np.mean(z)
                s2 = sum(x**2) / len(x)
                rho = sum(x * z) / len(x)
                slope = (rho - x_avg * z_avg) / (s2 - x_avg**2)
                velos[gene] = slope
                
        megs = []     
        pearsonr_qvals = fdrcorrection(pearsonr_pvals)[1]
        for ind, gene in enumerate(meg_candidates):
            if pearsonr_qvals[ind] > alpha:
                velos.pop(gene)
            else:
                megs.append(gene)
            
            
    pearsonr_pvals, pearsonr_qvals = np.array(pearsonr_pvals), np.array(pearsonr_qvals)
    pearsonr_pvals = pearsonr_pvals[pearsonr_qvals<=alpha]
    pearsonr_qvals = pearsonr_qvals[pearsonr_qvals<=alpha]
    
    sd.latent_z = zs_lat[megs]
    sd.megs = megs
    sd.pvals = pd.DataFrame(pearsonr_pvals, index=megs).T
    sd.qvals = pd.DataFrame(pearsonr_qvals, index=megs).T

    vels = []
    for gene in data.columns:
        if gene in velos:
            if not np.isnan(velos[gene]):
                vels.append(velos[gene])
            else:
                vels.append(0)
        else:
            vels.append(0)
    vels = np.array(vels)
    vels = np.clip(
        vels, np.quantile(vels[vels != 0], 0.05), np.quantile(vels[vels != 0], 0.95)
    )
    vels[~np.isin(data.columns, sd.megs)] = 0
    sd.velocity = vels

    return sd
