import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


plt.rc('text', usetex=True)

outpath = "../../plots/temperatures"
inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
T_n_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$']

# Draw Raw temperature plots
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(10, 5))
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	maxval = np.amax(T[:, i])
	minval = np.amin(T[:, i])
	dif = maxval - minval
	for pnt in range(0, np.size(N[:,0])):
		ax.scatter(N[pnt,0], T[pnt, i], marker='.', color=colors[Mode[pnt]])
	crushflag = np.invert(S[:, i])
	crushflag = crushflag.astype(int)
	ax.fill_between(N[:,0], minval, minval + dif * crushflag,
                facecolor='maroon',
                interpolate=True,
                zorder=0,
                alpha=0.2)
	plt.grid()
	plt.ylabel(T_n_labels[i])
	plt.xlabel(r'$n$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "T_{0}_raw.png".format(i+1)))
	plt.clf()

# Draw Raw temperature plots
for i in range(0, 10):
	fig, ax = plt.subplots(figsize=(10, 5))
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
	maxval = np.amax(T[:, i])
	minval = np.amin(T[:, i])
	dif = maxval - minval
	frm = np.size(N[:,0]) - 500
	for pnt in range(frm, np.size(N[:,0])):
		ax.scatter(N[pnt,0], T[pnt, i], marker='.', color=colors[Mode[pnt]])
	crushflag = np.invert(S[frm:, i])
	crushflag = crushflag.astype(int)
	ax.fill_between(N[frm:,0], minval, minval + dif * crushflag,
                facecolor='maroon',
                interpolate=True,
                zorder=0,
                alpha=0.5)
	plt.grid()
	plt.ylabel(T_n_labels[i])
	plt.xlabel(r'$n$')
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "T_{0}_raw_tail.png".format(i+1)))
	plt.clf()