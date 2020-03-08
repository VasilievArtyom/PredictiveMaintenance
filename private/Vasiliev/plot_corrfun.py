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
T_n_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$']


nlags=200
acf_vals = np.zeros((nlags, 10))
confit_vals = np.zeros((nlags, 2, 10))
for n in range(0,10):
	acf_vals[:, n], confit_vals[:, :, n] = acf(T[:, n], unbiased=True, nlags=nlags-1, qstat=False, alpha=.20, fft=False)
lags=np.arange(1, nlags+1, 1)


# Draw acf plots
outpath = "../../plots/acf"
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.fill_between(lags[1:], confit_vals[1:, 0, i], confit_vals[1:, 1, i], 
		facecolor='gainsboro', interpolate=True)
	ax.bar(lags[1:], acf_vals[1:, i], color='crimson')
	plt.grid()
	plt.ylabel(r'$r_{\tau}$'+' for '+T_n_labels[i])
	plt.xlabel(r'$\tau$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.draw()
	fig.savefig(path.join(outpath, "ACF_T_{0}.png".format(i+1)))
	plt.clf()

# Draw Pearson correlation coefficient (pcc) plot
outpath = "../../plots/pcc"
r = np.zeros((10, 10))
r_pvalues = np.zeros((10, 10))
for i in range(0, 10):
	for j in range(0, 10):
		r[i, j], r_pvalues[i, j] = stats.pearsonr(T[:, i], T[:, j])
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(r)
ax.set_xticks(np.arange(len(T_n_labels)))
ax.set_yticks(np.arange(len(T_n_labels)))

ax.set_xticklabels(T_n_labels)
ax.set_yticklabels(T_n_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(0, 10):
    for j in range(0, 10):
        text = ax.text(j, i, np.round(r[i, j],1),
                       ha="center", va="center", color="w")

ax.set_title("Pearson correlation coefficient")
fig.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "pcc.png"))
plt.clf()