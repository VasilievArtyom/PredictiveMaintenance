import math
import numpy as np
from numpy import *
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import scipy.stats as stats

plt.rc('text', usetex=True)

inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
B_n_labels = [r'$B_1$', r'$B_2$', r'$B_3$', r'$B_4$', r'$B_5$', r'$B_6$', r'$B_7$', r'$B_8$', r'$B_9$', r'$B_{10}$']


(unique, counts) = np.unique(Mode, return_counts=True)
print(unique)
print(counts)

# Draw acf plots
# outpath = "../../plots/mode"

# fig, ax = plt.subplots(figsize=(10, 5))
# ax.bar(N[:, 0], Mode, color='crimson')
# plt.grid(which='both', axis='both')
# plt.ylabel(r'Mode')

# plt.xlabel(r'n')
# plt.tight_layout()
# plt.draw()
# plt.show()
# # fig.savefig(path.join(outpath, "mode_in_time.png"))
# plt.clf()

