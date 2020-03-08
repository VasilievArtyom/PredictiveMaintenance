import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


plt.rc('text', usetex=True)

outpath = "../../plots/acf"
inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


nlags=200
acf_vals = np.zeros((nlags, 10))
confit_vals = np.zeros((nlags, 2, 10))
for n in range(0,10):
	acf_vals[:, n], confit_vals[:, :, n] = acf(T[:, n], unbiased=True, nlags=nlags-1, qstat=False, alpha=.05, fft=False)
lags=np.arange(1, nlags+1, 1)


# Draw acf plots
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(10, 5))
	ax.fill_between(lags[1:], confit_vals[1:, 0, i], confit_vals[1:, 1, i], 
		facecolor='gainsboro', interpolate=True)
	ax.bar(lags[1:], acf_vals[1:, i], color='crimson')
	plt.grid()
	plt.ylabel(r'$r_{\tau}$'+' for T'+str(i))
	plt.xlabel(r'$n$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.draw()
	fig.savefig(path.join(outpath, "ACF_T_{0}.png".format(i+1)))
	plt.clf()

# # Draw QLB p-values plot
# fig, ax = plt.subplots(figsize=(4, 3.8))
# ax.bar(lags[1:-1], pvalues[1:], color='crimson')
# plt.grid()
# # plt.ylabel(r'Ljung–Box Q test p-values')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# plt.title(r'Ljung–Box Q test p-values')
# #ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "qlb200.png"))
# plt.clf()