import os

import matplotlib.pyplot as plt
import numpy as np

from gillespie_scrna_variable_paras import Gillespie
from sim_utils import *

"""
    init expression programs
"""



num_elements = 5
system = Gillespie(
    num_elements,
    inits=[1, 0, 0, 0, 0],
    max_cell_num=8000
)

## 0 -> 1 -> 2 -> 3 -> 4
p0 = lambda t: (1 - 1 / (1 + np.exp(-0.6 * (t - 17))))
p1 = lambda t: (1 - 1 / (1 + np.exp(-0.6 * (t - 17))))
p2 = lambda t: (1 - 1 / (1 + np.exp(-0.6 * (t - 17))))
p3 = lambda t: (1 - 1 / (1 + np.exp(-0.6 * (t - 18))))
p4 = lambda t: (1 - 1 / (1 + np.exp(-0.6 * (t - 19))))
d0 = lambda t: 1 - p0(t)
d1 = lambda t: 1 - p1(t)
d2 = lambda t: 1 - p2(t)
d3 = lambda t: 1 - p3(t)

system.add_reaction(p0, [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], index=0)
system.add_reaction(p1, [0, 1, 0, 0, 0], [0, 2, 0, 0, 0], index=1)
system.add_reaction(p2, [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], index=2)
system.add_reaction(p3, [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], index=3)
system.add_reaction(p4, [0, 0, 0, 0, 1], [0, 0, 0, 0, 2], index=4)
system.add_reaction(d0, [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], index=5)
system.add_reaction(d1, [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], index=6)
system.add_reaction(d2, [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], index=7)
system.add_reaction(d3, [0, 0, 0, 1, 0], [0, 0, 0, 0, 1], index=8)


system.evolute(20000000)
t = np.array(system.generation_time)
cell_num_traj = np.array(system.n)

c0 = cell_num_traj[:, 0]
c1 = cell_num_traj[:, 1]
c2 = cell_num_traj[:, 2]
c3 = cell_num_traj[:, 3]
c4 = cell_num_traj[:, 4]
plt.plot(t, c0, c="tab:gray", label="0")
plt.plot(t, c1, c="tab:blue", label="1")
plt.plot(t, c2, c="tab:green", label="2")
plt.plot(t, c3, c="tab:red", label="3")
plt.plot(t, c4, c="tab:orange", label="4")
plt.legend()
plt.show()

"""
    write to file
"""
print("writing to file...\n")
data_path = "../datas/"
tree_file_name = "tree_origin_linear.csv0"
cell_num_file_name = "cell_num_linear.csv0"

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
