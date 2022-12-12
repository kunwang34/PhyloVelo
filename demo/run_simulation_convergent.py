import os

import matplotlib.pyplot as plt
import numpy as np

from gillespie_scrna_variable_paras import Gillespie
from sim_utils import *

"""
    init expression programs
"""


num_elements = 6

system = Gillespie(
    num_elements,
    inits=[1, 0, 0, 0, 0, 0],
    max_cell_num=6000,
)


r0 = 1

p0 = lambda t: r0 * (1 - 1 / (1 + np.exp(-0.6 * (t - 17.7))))
p12 = lambda t: r0 * (1 - 1 / (1 + np.exp(-0.6 * (t - 16))))
p34 = lambda t: r0 * (1 - 1 / (1 + np.exp(-0.6 * (t - 16))))
p5 = lambda t: r0 * (1 - 1 / (1 + np.exp(-0.6 * (t - 16))))

d0 = lambda t: 1 - p0(t)
d12 = lambda t: 1 - p12(t)
d34 = lambda t: 1 - p34(t)

system.add_reaction(p0, [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], index=0)
system.add_reaction(p12, [0, 1, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0], index=1)
system.add_reaction(p12, [0, 0, 1, 0, 0, 0], [0, 0, 2, 0, 0, 0], index=2)
system.add_reaction(p34, [0, 0, 0, 1, 0, 0], [0, 0, 0, 2, 0, 0], index=3)
system.add_reaction(p34, [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 2, 0], index=4)
system.add_reaction(p5, [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 2], index=5)

system.add_reaction(d0, [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], index=6)
system.add_reaction(d0, [1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], index=7)
system.add_reaction(d12, [0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], index=8)
system.add_reaction(d12, [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0], index=9)
system.add_reaction(d34, [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], index=10)
system.add_reaction(d34, [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], index=11)

system.evolute(20000000)
t = np.array(system.generation_time)
cell_num_traj = np.array(system.n)

c0 = cell_num_traj[:, 0]
c1 = cell_num_traj[:, 1]
c2 = cell_num_traj[:, 2]
c3 = cell_num_traj[:, 3]
c4 = cell_num_traj[:, 4]
c5 = cell_num_traj[:, 5]

plt.plot(t, c0, c="tab:gray", label="0")
plt.plot(t, c1, c="tab:blue", label="1")
plt.plot(t, c2, c="tab:green", label="2")
plt.plot(t, c3, c="tab:red", label="3")
plt.plot(t, c4, c="tab:orange", label="4")
plt.plot(t, c4, c="tab:purple", label="5")

plt.legend()
plt.show()

"""
    write to file
"""

print("writing to file...\n")
data_path = "../datas/"
tree_file_name = "tree_origin_convergent.csv0"
cell_num_file_name = "cell_num_convergent.csv0"

curr_cells = []

for i in system.curr_cells.values():
    curr_cells += i

while tree_file_name in os.listdir(data_path):
    tree_file_name = tree_file_name[:-1] + str(int(tree_file_name[-1]) + 1)
    cell_num_file_name = cell_num_file_name[:-1] + str(int(cell_num_file_name[-1]) + 1)

np.savetxt(
    data_path + cell_num_file_name,
    np.hstack((t.reshape(-1, 1), cell_num_traj)),
    fmt="%.5f",
)

wirte_lineage_info(
    data_path + tree_file_name, system.anc_cells, curr_cells, system.t[-1]
)


print("done")
