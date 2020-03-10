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


failure = np.sum(np.invert(S).astype(int), axis=0)


# Draw acf plots
outpath = "../../plots/stat"

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(np.arange(10), failure, color='crimson')
plt.grid(which='both', axis='y')
plt.ylabel(r'Failures per block count in time samples')

ax.set_xticks(np.arange(len(B_n_labels)))
ax.set_xticklabels(B_n_labels)
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "FailurePerBlock.png"))
plt.clf()

