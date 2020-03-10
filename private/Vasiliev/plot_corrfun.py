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





# Draw acf plots
outpath = "../../plots/acf"
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(8, 3.8))
	ax.fill_between(lags[1:], confit_vals[1:, 0, i], confit_vals[1:, 1, i], 
		facecolor='gainsboro', interpolate=True)
	ax.bar(lags[1:], acf_vals[1:, i], color='crimson')
	plt.grid()
	plt.ylabel(r'$r_{\tau}$'+' for '+T_n_labels[i])
	plt.xlabel(r'$\tau$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "ACF_T_{0}.png".format(i+1)))
	plt.clf()
