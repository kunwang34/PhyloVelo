from collections import defaultdict
from copy import deepcopy
from math import log
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import numpy as np
import pandas as pd
import scipy as sp
from scipy import stats
from scipy.special import factorial, comb
from scipy.stats import nbinom

from .gene_expr import *
import os



def get_annotation(file):
    '''
    Get simulation data annotation 
    
    Args:
        file:
            Tree file from simulation script
    
    Return:
        list:
            Cell names
        list:
            Cell states
        list:
            Cell generations
    '''
    annotation = pd.read_csv(file)
    cell_names = []
    cell_states = []
    cell_generation = []
    for i in range(annotation.shape[0]):
        cell_generation.append(int(annotation.loc[i].generation))
        cell_names.append(
            f"<{int(annotation.loc[i].generation)}_{int(annotation.loc[i].cell_id)}>"
        )
        cell_states.append(int(annotation.loc[i].state))
    return cell_names, cell_states, cell_generation


def sim_base_expr(
    tree: "bio.phylo.tree",
    cell_states: "pd.DataFrame",
    Ngene: int,
    r_variant_gene: float,
    diff_map: dict,
    forward_map: dict = {},
    mu0_loc=20,
    mu0_scale=3,
    drift_loc=0,
    drift_scale=1,
    pseudo_state_time: dict = None,
):
    '''
    Simulation base expression
    
    Args:
        tree:
            Phylogenetic tree
        cell_states:
            DataFrame of cell types with index of cell names
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
        mu0_loc:
            Mean of initial expression
        mu0_scale:
            Variation of initial expression
        drift_loc:
            Mean of gene drift
        drift_scale:
            Variation of drift
    Returns:
        class:
            Gene expr program
        pd.DataFrame:
            base expression matrix
    '''
    base_expr = pd.DataFrame()
    depth = defaultdict(list)
    Nstate = len(diff_map)
    terminals_depths = tree.depths()
    for i in tree.get_terminals():
        depth[int(cell_states.loc[i.name])].append(int(terminals_depths[i]))

    start_time = {}
    end_time = {}
    for i in range(Nstate):
        start_time[i] = int(np.quantile(depth[i], 0.2))
        end_time[i] = int(np.quantile(depth[i], 0.8))

    t0 = start_time[0]
    for i in start_time:
        start_time[i] -= t0
        end_time[i] -= t0

    state_time = {}
    for i in range(Nstate):
        state_time[i] = [start_time[i], end_time[i]]

    pseudo_end_time = {0: end_time[0]}
    pseudo_start_time = {0: 0}
    for i in range(1, Nstate):
        pseudo_start_time[i] = (
            int(np.mean([pseudo_end_time[anc] for anc in diff_map[i]])) + 2
        )
        pseudo_end_time[i] = end_time[i] - start_time[i] + pseudo_start_time[i]

    if pseudo_state_time is None:
        pseudo_state_time = {}
        for i in range(Nstate):
            if not i in forward_map.values():
                pseudo_state_time[i] = [pseudo_start_time[i], pseudo_end_time[i]]
            else:
                pseudo_state_time[i] = [
                    pseudo_state_time[diff_map[i][0]][0] + 2,
                    pseudo_state_time[diff_map[i][0]][1] + 2,
                ]
    else:
        pseudo_start_time = {}
        pseudo_end_time = {}
        for i in pseudo_state_time:
            pseudo_start_time[i] = pseudo_state_time[i][0]
            pseudo_end_time = pseudo_state_time[i][1]

    ge = GeneExpr(
        Ngene=Ngene,
        r_variant_gene=r_variant_gene,
        diff_map=diff_map,
        forward_map=forward_map,
        state_time=pseudo_state_time,
    )

    ge.generate_genes(mu0_loc, mu0_scale, drift_loc, drift_scale)
    for cell in tree.get_terminals():
        cellstate = int(cell_states.loc[cell.name])
        base_expr[cell.name] = ge.expr(
            cellstate,
            int(terminals_depths[cell])
            - t0
            - start_time[cellstate]
            + pseudo_start_time[cellstate],
        )
    base_expr = base_expr.T
    return ge, base_expr


def add_lineage_noise(
    tree: "bio.phylo.tree", base_expr_mat: "pd.DataFrame", scale=0.0001
):
    '''
    Simulate lineage noise
    
    Args:
        tree:
            Phylogenetic tree
        base_expr_mat:
            Base expression matrix from sim_base_expr
        scale:
            Lineage noise scale
    
    Return:
        pd.DataFrame:
            Base expression matrix with lineage noise
    
    '''
    noise = dict()
    base_expr_mat = deepcopy(base_expr_mat)
    ngene = base_expr_mat.shape[1]
    for cl in tree.get_terminals():
        path = tree.get_path(cl)
        noise[path[0].name] = np.zeros(ngene)
        for i, anc in enumerate(path):
            if not anc.name in noise:
                noise[anc.name] = np.random.normal(
                    loc=noise[path[i - 1].name], scale=scale
                )
    for i in base_expr_mat.index:
        base_expr_mat.loc[i] += noise[i]
    base_expr_mat = base_expr_mat.clip(lower=0)
    return base_expr_mat


def get_count_from_base_expr(base_expr_mat: "pd.DataFrame", alpha: int = 3):
    '''
    Draw gene expression count from base expression matrix
    
    Args:
        base_expr_mat:
            Base expression matrix
        alpha:
            Scale parameter of NB distribution
    
    Return:
        Gene count matrix
    '''
    base_expr_mat = deepcopy(base_expr_mat)
    for i in range(base_expr_mat.shape[0]):
        mu = base_expr_mat.iloc[i, :]
        mu = np.clip(mu, a_min=1e-5, a_max=None)
        sigma2 = mu + alpha * mu**2
        p, r = mu / (sigma2), mu**2 / (sigma2 - mu)
        base_expr_mat.iloc[i, :] = nbinom(r, p).rvs()
    return base_expr_mat


def get_count(paras:list):
    '''
    Draw random sample form NB distribution with paras = (r, p)
    
    Args:
        paras:
            NB parameters, [(r,p)]
    
    Return:
        int:
            Random sample
    '''
    r, p = [], []
    for para in paras:
        r.append(para[0])
        p.append(para[1] if para[1] >= 0 else 1)
    return stats.nbinom(r, p).rvs()


def reconstruct(file:str, output:str=None, seed:int=None, is_balance:bool=False, **kwargs):
    """
    Reconstruct phylogenetic tree of simulation data
    
    Args:
        file:
            Simulation file path
        output:
            Output newick file path
        seed:
            Random seed
        is_balance:
            Is all cell types' cell number equal
        ratio:
            How many cells to reconstruct
    Return:
        newick tree at output file
    """
    if seed:
        np.random.seed(seed)
    data = pd.read_csv(file)
    alives = data[data.is_alive == 1]

    if "ratio" in kwargs:
        sample_num = int(kwargs["ratio"] * alives.shape[0])
    elif "num" in kwargs:
        sample_num = kwargs["num"]
    else:
        warnings.warn("No ratio or num given, default plot all tree")
        sample_num = int(alives.shape[0])

    if is_balance:
        sample_index = np.array([])
        states = set(alives.state.to_numpy())
        for i in states:
            index = alives[alives.state == i].index.to_numpy()
            sample_index = np.concatenate(
                (
                    sample_index,
                    np.random.choice(
                        index, int(sample_num / len(states)), replace=False
                    ),
                )
            )

    else:
        index = alives.index.to_numpy()
        sample_index = np.random.choice(index, sample_num, replace=False)
    sample_index = list(sample_index)
    ##    print(data.loc[sample_index])

    new_keep = list(sample_index.copy())
    while new_keep:
        new_parents = []
        for i in new_keep:
            new_parents.append(
                data[
                    (data.generation == int(data.loc[i].generation) - 1)
                    & (data.cell_id == int(data.loc[i].parent_id))
                ].index[0]
            )
        while 0 in new_parents:
            new_parents.remove(0)
        new_keep = deepcopy(new_parents)
        sample_index.extend(new_parents)
    sample_index = np.unique(sample_index)
    sample_index = np.insert(sample_index, 0, 0)
    data = data.loc[sample_index]

    data["info"] = [
        "<{:d}_{:d}>:{}".format(
            int(data.loc[i]["generation"]),
            int(data.loc[i]["cell_id"]),
            data.loc[i]["time_death"] - data.loc[i]["time_birth"],
        )
        for i in data.index
    ]

    states = [
        "<{:d}_{:d}>:{:d}".format(
            int(data.loc[i]["generation"]),
            int(data.loc[i]["cell_id"]),
            int(data.loc[i]["state"]),
        )
        for i in data.index
    ]
    gen = data.generation.max()
    tree = []

    while gen:
        for pid in set(data[data.generation == gen].parent_id.to_numpy()):
            pair = data[
                np.all(list(zip(data.generation == gen, data.parent_id == pid)), axis=1)
            ]
            parent_index = data[
                np.all(
                    list(zip(data.generation == gen - 1, data.cell_id == pid)), axis=1
                )
            ].index[0]
            oi = data.loc[parent_index, "info"]
            if pair.shape[0] == 2:
                ni = "({}, {}){}".format(pair.iloc[0]["info"], pair.iloc[1]["info"], oi)
            else:
                ni = "({}){}".format(pair.iloc[0]["info"], oi)
            data.loc[parent_index, "info"] = ni
            data = data.drop(index=pair.index)
        gen -= 1

    with open(output, "w") as f:
        f.write(data.loc[0, "info"])
        f.write("\n")
        for i in states:
            f.write(i)
            f.write("\t")


def wirte_lineage_info(filepath, anc_cells, curr_cells, curr_time):
    '''
    Record lineage infomation in simulation
    '''
    with open(filepath, mode="w") as f:
        f.write(
            ",".join(
                [
                    "generation",
                    "cell_id",
                    "parent_id",
                    "state",
                    "is_alive",
                    "time_birth",
                    "time_death",
                ]
            )
        )
        f.write("\n")
        for c in anc_cells:
            f.write(
                ",".join(
                    [
                        str(c.gen),
                        str(c.cid),
                        str(c.parent),
                        str(c.state),
                        "0",
                        str(c.tb),
                        str(c.td),
                    ]
                )
            )
            f.write("\n")

        for c in curr_cells:
            f.write(
                ",".join(
                    [
                        str(c.gen),
                        str(c.cid),
                        str(c.parent),
                        str(c.state),
                        "1",
                        str(c.tb),
                        str(curr_time),
                    ]
                )
            )
            f.write("\n")


class Cell:
    """
    Cell class
    
    Args:
        Ngene: 
            Gene number
        state: 
            Cell type
        gen: 
            Cell generation
        cid: 
            Cell id
        parent: 
            Cell's parent
        tb: 
            Birth time
        td: 
            Death time
    """
    def __init__(
        self,
        Ngene:int = None,
        state:int = 0,
        gen:int = None,
        cid:int = None,
        parent:int = None,
        tb:float = None,
        td:float = None,
    ):
        self.state = state
        self.parent = parent
        self.cid = cid
        self.gen = gen
        self.tb = tb
        self.td = td


class Reaction:
    '''
    Cell division/differentiation type
    
    Args:
        rate:
            reaction rate function
        num_lefts:
            Cell numbers before reaction
        num_right:
            Cell numbers after reaction
        index:
            Reaction index
    '''
    def __init__(self, rate:callable=None, num_lefts:list=None, num_rights:list=None, index:int=None):
        self.rate = rate
        assert len(num_lefts) == len(num_rights)
        self.num_lefts = np.array(num_lefts)
        self.num_rights = np.array(num_rights)
        self.num_diff = self.num_rights - self.num_lefts
        self.index = index
        if 2 * sum(num_lefts) == sum(num_rights):
            self.type = "proliferate"
        else:
            self.type = "differentiation"

    def combine(self, n, s):
        return np.prod(comb(n, s))

    def propensity(self, n, t):
        return self.rate(t) * self.combine(n, self.num_lefts)


class Gillespie:
    '''
    Gillespie simulation
    
    Args:
        num_elements: 
            Cell type number
        inits: 
            Initial cell number
        max_cell_num: 
            Maximum cell number
    '''
    def __init__(
        self,
        num_elements:int,
        inits:list=None,
        max_cell_num:int=20000,
    ):

        assert num_elements > 0
        self.num_elements = num_elements
        self.reactions = []
        if inits is None:
            self.n = [np.ones(self.num_elements)]
        else:
            self.n = [np.array(inits)]
        self.anc_cells = []
        self.curr_cells = defaultdict(list)
        self.curr_cells[0] = [
            Cell(gen=0, parent=-1, cid=i, tb=0) for i in range(int(self.n[0][0]))
        ]

        self.generation_time = [0]
        self.max_cell_num = max_cell_num

    def add_reaction(self, rate:callable=None, num_lefts:list=None, num_rights:list=None, index:int=None):
        '''
        Add reactions to simulation
        
        Args:
            rate:
                reaction rate function
            num_lefts:
                Cell numbers before reaction
            num_right:
                Cell numbers after reaction
            index:
                Reaction index
        '''
        assert len(num_lefts) == self.num_elements
        assert len(num_rights) == self.num_elements
        self.reactions.append(Reaction(rate, num_lefts, num_rights, index))

    def evolute(self, steps:int):
        '''
        Run simulation
        
        Args:
            steps:
                How many steps to evolute before step
        '''
        self.t = [0]
        self.log = []
        cell_num_per_gen = {0: self.n[0][0]}
        cell_id_per_gen = {0: self.n[0][0]}

        def proliferate(i):
            node = np.random.choice(self.curr_cells[i])
            l1, l2 = deepcopy(node), deepcopy(node)
            l1.parent, l2.parent = node.cid, node.cid
            l1.gen += 1
            l2.gen += 1
            l1.tb, l2.tb = self.t[-1], self.t[-1]
            try:
                index = cell_id_per_gen[l1.gen] + 1
            except:
                index = 0
                cell_id_per_gen.update({l1.gen: -1})
                cell_num_per_gen.update({l1.gen: -1})

            l1.cid, l2.cid = index, index + 1
            self.curr_cells[i].append(l1)
            self.curr_cells[i].append(l2)

            cell_num_per_gen[l1.gen] += 2
            cell_id_per_gen[l1.gen] += 2
            cell_num_per_gen[l1.gen - 1] -= 1

            self.curr_cells[i].remove(node)
            node.td = self.t[-1]
            self.anc_cells.append(node)

        def differentiate(i, j):

            node = np.random.choice(self.curr_cells[i])
            diffcell = deepcopy(node)
            self.curr_cells[i].remove(node)

            node.td = self.t[-1]
            diffcell.tb = self.t[-1]
            self.anc_cells.append(node)
            diffcell.gen += 1

            diffcell.state = j

            try:
                index = cell_id_per_gen[diffcell.gen] + 1
            except:
                index = 0
                cell_id_per_gen.update({diffcell.gen: -1})
                cell_num_per_gen.update({diffcell.gen: -1})

            diffcell.cid = index
            cell_id_per_gen[diffcell.gen] += 1
            cell_num_per_gen[diffcell.gen] += 1
            cell_num_per_gen[diffcell.gen - 1] -= 1
            diffcell.parent = node.cid
            self.curr_cells[diffcell.state].append(diffcell)

        class SwitchCase(object):
            def case_to_func(self, reaction):
                self.reaction = reaction
                method = getattr(self, reaction.type + "1")
                return method

            def proliferate1(self):
                proliferate(list(self.reaction.num_lefts).index(1))

            def differentiation1(self):
                differentiate(
                    list(self.reaction.num_lefts).index(1),
                    list(self.reaction.num_rights).index(1),
                )

        cls = SwitchCase()

        with tqdm(total=self.max_cell_num) as pbar:

            for _ in range(steps):
                all_cell_num = sum(cell_num_per_gen.values())
                pbar.update(all_cell_num - pbar.n)
                if all_cell_num > self.max_cell_num:
                    print("\n maximum cell number reached")
                    break

                avg_generation = np.dot(
                    list(cell_num_per_gen.keys()), list(cell_num_per_gen.values())
                ) / sum(cell_num_per_gen.values())

                self.generation_time.append(avg_generation)
                A = np.array(
                    [
                        rec.propensity(self.n[-1], avg_generation)
                        for rec in self.reactions
                    ]
                )

                A0 = A.sum()
                A /= A0
                t0 = -np.log(np.random.random()) / A0
                self.t.append(self.t[-1] + t0)
                react = np.random.choice(self.reactions, p=A)

                self.log.append(react.index)
                self.n.append(self.n[-1] + react.num_diff)

                cls.case_to_func(react)()
