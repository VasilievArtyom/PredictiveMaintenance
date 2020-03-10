import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from scipy.optimize import minimize
import scipy.stats as stats

plt.rc('text', usetex=True)

# outpath = "../../plots/positions"
inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

def pos_functional(_x, _r):
	stat_value = 0.0;
	for i in range(0, 10):
		for j in range(0, i):
			stat_value += np.abs( (_x[i] - _x[j])**2 + (_x[i + 10] - _x[10 + j])**2 - np.exp(((1 - _r[i, j]))**2) )
	return stat_value

r = np.zeros((10, 10))
r_pvalues = np.zeros((10, 10))
for i in range(0, 10):
	for j in range(0, 10):
		r[i, j], r_pvalues[i, j] = stats.pearsonr(T[:, i], T[:, j])

n_iterations = 1
x = np.zeros((20, n_iterations))
for it in range (0, n_iterations):
	x0 = np.random.rand(20,)
	res = minimize(pos_functional, x0, args=r, method='nelder-mead')
	x[:, it] = res.x
x_mean = np.mean(x, axis=1)
x_mean_err = np.zeros(20)
for tmp in range(0, 20):
	x_mean_err[tmp] = np.std(x[tmp, : ])

B_n_labels = [r'$B_1$', r'$B_2$', r'$B_3$', r'$B_4$', r'$B_5$', r'$B_6$', r'$B_7$', r'$B_8$', r'$B_9$', r'$B_{10}$']

outpath = ""
x = x_mean[0:10]
y = x_mean[10:20]
x_err = x_mean_err[0:10]
y_err = x_mean_err[10:20]
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(x, y, yerr=y_err, xerr=x_err, fmt='o', marker='o', capsize=5, capthick=1, ecolor='black', color='r', alpha=0.3)
plt.grid()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
for blccnt in range(0, 10):
	plt.annotate(B_n_labels[blccnt], (x[blccnt], y[blccnt]), color='blue')
# plt.title(r'$title$')
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.draw()
fig.savefig(path.join(outpath, "blocks_poss_guess.png"))