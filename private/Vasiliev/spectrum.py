import math
import numpy as np
from numpy import *
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import scipy.stats as stats
from scipy import signal

plt.rc('text', usetex=True)

inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
T_n_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$']


# Draw acf plots
outpath = "../../plots/psd"
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(10, 5))
	f, Pxx_den = signal.periodogram(T[:, i] - np.average(T[:, i]), window='hamming')
	ax.semilogy(f, Pxx_den, color='crimson')
	plt.grid()
	plt.ylabel(r'$\frac{1}{T} {|X_{T} (i \omega)|}^2 $')
	plt.xlabel(r'$\omega$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.title(r'Spectral density estimation for '+T_n_labels[i])
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "psd_T_{0}.png".format(i+1)))
	plt.clf()
