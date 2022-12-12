import os
# from math import *
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import beta, gamma, nbinom, norm, uniform


class Gene:
    '''
    Gene class
    
    Args:
        mu0:
            Initial expression
        drift:
            Drift coefficient of DP
        sigma:
            Diffusion coefficient of DP
        t0:
            Gene initial time
    '''
    def __init__(self, mu0:float, drift:float, sigma:float=None, t0:int=0):
        self.mu0 = mu0
        self.drift = drift
        if sigma:
            self.sigma = sigma
        else:
            self.sigma = max(
                0.00001, norm(loc=mu0 / 100, scale=max(mu0 / 300, 0.001)).rvs()
            )
        self.t0 = t0
        self.base_expr = [mu0]

    def diffusion(self):
        '''
        Diffusion one step.
        '''
        self.base_expr.append(
            norm(loc=self.base_expr[-1] + self.drift, scale=self.sigma).rvs()
        )

    def base_expr_calc(self, t:int):
        '''
        Calculate base expression
        
        Args
            t:
                time
        Return:
            Base expression at time t
        '''
        t = max(0, t - self.t0)
        while len(self.base_expr) <= t:
            self.diffusion()
        return max(self.base_expr[t], 0)


class GeneExpr:
    '''
    Gene expression program
    
    Args:
        Ngene:
            Gene number
        r_variant_gene:
            Ratio of gene changes with differentiation
        diff_map:
            Differentiation relationships between different cell types
            {a:[b,c]} means 'a' is differentiated from 'b' and 'c'
        state_time:
            Pseudo time of each states
        forward_map:
            Only use in convergent model simulation
            {a:b} means 'a' will differentiated to 'b'
    '''
    def __init__(
        self,
        Ngene: int,
        r_variant_gene: float,
        diff_map: dict,
        state_time: dict,
        forward_map: dict = None,
    ):
        assert r_variant_gene <= 1

        self.Ngene = Ngene
        Nvariant_gene = int(r_variant_gene * Ngene)
        Ninvariant_gene = Ngene - Nvariant_gene

        self.Nstates = max(diff_map.keys()) + 1

        cells = defaultdict(list)
        cells[0] = [0] * Ngene
        self.variant_gene = {}
        self.variant_gene[0] = set()
        for i in range(1, self.Nstates):
            variant_gene = set(
                np.random.choice(range(Ngene), Nvariant_gene, replace=False)
            )
            self.variant_gene[i] = variant_gene
            for gene in range(Ngene):
                if gene in variant_gene:
                    cells[i].append(i)
                else:
                    cells[i].append(np.random.choice(diff_map[i]))

        self.expr_rec = defaultdict(dict)
        self.cells_genetype = cells
        self.diff_map = diff_map
        self.state_time = state_time
        self.forward_map = forward_map

    def generate_genes(self, mu0_loc:float=20, mu0_scale:float=3, drift_loc:float=0, drift_scale:float=1):
        '''
        Generate genes
        
        Args:
            mu0_loc:
                Mean of initial expression
            mu0_scale:
                Variation of initial expression
            drift_loc:
                Mean of gene drift
            drift_scale:
                Variation of drift
        '''
        cells = deepcopy(self.cells_genetype)
        cells[0] = [
            Gene(
                mu0=max(0, norm(loc=mu0_loc, scale=mu0_scale).rvs()),
                drift=norm(loc=drift_loc, scale=drift_scale).rvs(),
            )
            for _ in range(self.Ngene)
        ]
        # cells[0] = [Gene(mu0=gamma(a=0.5, scale=20).rvs(), drift=scale*(beta(0.5,0.5).rvs()-0.5)) for _ in range(self.Ngene)]
        for state in range(1, self.Nstates):
            for gene in range(self.Ngene):
                if len(self.diff_map[state]) > 1:
                    is_same_to_anc = [
                        self.cells_genetype[state][gene] == self.cells_genetype[i][gene]
                        for i in self.diff_map[state]
                    ]
                    any_same = 0
                    for ind, is_same in enumerate(is_same_to_anc):
                        if is_same:
                            cells[state][gene] = cells[self.diff_map[state][ind]][gene]
                            any_same = 1
                            break
                    if not any_same:
                        anc = np.random.choice(self.diff_map[state])
                        cells[state][gene] = Gene(
                            mu0=cells[anc][gene].base_expr_calc(
                                self.state_time[state][0]
                            ),
                            drift=np.random.choice([-1, 1]) * cells[anc][gene].drift,
                            t0=self.state_time[state][0],
                        )

                else:
                    if (
                        self.cells_genetype[state][gene]
                        == self.cells_genetype[self.diff_map[state][0]][gene]
                    ):
                        ##                    cells[state][gene] = deepcopy(cells[self.diff_map[state]][gene])
                        ##                    cells[state][gene].t0 = self.state_time[state][0]
                        cells[state][gene] = cells[self.diff_map[state][0]][gene]

                    else:
                        cells[state][gene] = Gene(
                            mu0=cells[self.diff_map[state][0]][gene].base_expr_calc(
                                self.state_time[state][0]
                            ),
                            drift=np.random.choice([-1, 1])
                            * cells[self.diff_map[state][0]][gene].drift,
                            t0=self.state_time[state][0],
                        )

        for state in range(1, self.Nstates):
            if state in self.forward_map:
                next_state = self.forward_map[state]
                for gene in range(self.Ngene):
                    # if gene in self.variant_gene[next_state]:
                    cells[state][gene].drift = (
                        cells[next_state][gene].mu0 - cells[state][gene].mu0
                    ) / (self.state_time[next_state][0] - self.state_time[state][0])

        self.cells = cells

    def expr(self, state, time):
        ##        time = min(self.state_time[state][1], time)
        ##        time = time - self.state_time[state][0]
        time = np.clip(time, self.state_time[state][0], self.state_time[state][1])

        if not time in self.expr_rec[state]:
            self.expr_rec[state][time] = [
                i.base_expr_calc(time) for i in self.cells[state]
            ]
        return self.expr_rec[state][time]
