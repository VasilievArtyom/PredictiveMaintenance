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
outpath = "../../plots/mode"


colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
for currentMode in range(0, 5):
	fig, ax = plt.subplots(figsize=(2, 5))
	Tvals = np.zeros(10)
	difTvals = np.zeros(10)
	# print("######################################################")
	# print(currentMode)
	# print("######################################################")
	for i in range(0, (np.size(Mode))-1): 
		if (currentMode == Mode[i]):
			Tvals = np.vstack((Tvals, T[i,:]))
			# print(difTvals)
			difTvals = np.vstack((difTvals, T[i+1,:] - T[i,:]))
			# print(currentMode, np.round(np.abs(T[i+1,0] - T[i,0]), decimals=1))
			# print(currentMode, np.round(Tvals[:, 0], decimals=1))
	# print(currentMode, np.round(Tvals[:, 0], decimals=1))
	difTvals = difTvals[1:,:]
	Tvals = Tvals[1:,:]
	# print(currentMode, np.round(Tvals[:, 0], decimals=1))
	for block in range(0, 10):
		ax.scatter(difTvals[block,:], Tvals[block,:], marker='o', color=colors[block])
	plt.grid()
	plt.title(r'Mode='+str(currentMode))
	plt.ylabel(r'$T_n$')
	plt.xlabel(r'$\triangle T_n$')
	plt.ylim(20, 70)
	plt.xlim(-8, 8)
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "deltaToverTforMode_{0}.png".format(currentMode)))
	plt.clf()


for currentMode in range(0, 5):
	fig, ax = plt.subplots(figsize=(2, 5))
	Tvals = np.zeros(10)
	difTvals = np.zeros(10)
	for i in range(0, (np.size(Mode)-1)): 
		if (currentMode == Mode[i]):
			Tvals = np.vstack((Tvals, T[i,:]))
			difTvals = np.vstack((difTvals, T[i+1,:] - T[i,:]))
	difTvals = difTvals[1:,:]
	Tvals = Tvals[1:,:]

	avrgTvals = np.mean(Tvals, axis=1)
	avrgdifTvals = np.mean(difTvals, axis=1)

	avrgTvalsErr = np.zeros(np.size(Tvals[:,0]))
	avrgdifTvalsErr = np.zeros(np.size(difTvals[:,0]))

	for tmp in range(0, np.size(Tvals[:,0])):
		avrgTvalsErr[tmp] = np.std(Tvals[tmp, :])
		avrgdifTvalsErr[tmp] = np.std(difTvals[tmp, :])
	ax.errorbar(avrgdifTvals, avrgTvals, xerr=avrgdifTvalsErr, yerr=avrgdifTvalsErr, fmt='.', marker='.', capsize=5, capthick=1, ecolor='black', color='r')
	plt.grid()
	plt.title(r'Mode='+str(currentMode))
	plt.ylabel(r'$T_n$')
	plt.xlabel(r'$\triangle T_n$')
	plt.ylim(20, 70)
	plt.xlim(-5, 15)
	ax.xaxis.grid(b=True, which='both')
	ax.yaxis.grid(b=True, which='both')
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, "avrgdeltaToverTforMode_{0}.png".format(currentMode)))
	plt.clf()