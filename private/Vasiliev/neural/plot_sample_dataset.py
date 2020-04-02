import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import os
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)

inpath = "data/"
outpath = ""

currentfile = "data_T_0"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_ ),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('ma2T', np.float_, (10,)),
                        ('ma3T', np.float_, (10,)),
                        ('ma5T', np.float_, (10,)),
                        ('ma8T', np.float_, (10,)),
                        ('ma13T', np.float_, (10,))])
N, Mode, T, kalmanT, ma2T, ma3T, ma5T, ma8T, ma13T = np.loadtxt(path.join(inpath, currentfile + ".csv"), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

print(np.amax(T), np.amin(T), np.mean(T))
print(np.amax(kalmanT), np.amin(kalmanT), np.mean(kalmanT))
print(np.amax(ma2T), np.amin(ma2T), np.mean(ma2T))
print(np.amax(ma3T), np.amin(ma3T), np.mean(ma3T))
print(np.amax(ma5T), np.amin(ma5T), np.mean(ma5T))
print(np.amax(ma8T), np.amin(ma8T), np.mean(ma8T))
print(np.amax(ma13T), np.amin(ma13T), np.mean(ma13T))
print(np.amax(Mode), np.amin(Mode), np.mean(Mode))

B_n_labels = [r'$B_1$', r'$B_2$', r'$B_3$', r'$B_4$', r'$B_5$', r'$B_6$', r'$B_7$', r'$B_8$', r'$B_9$', r'$B_{10}$']
print_last_tmstms = 100
fontsize = 13
fig, ax = plt.subplots(nrows=10, ncols=7, constrained_layout=True, figsize=(10, 15))
for blc_id in range(0, 10):
	ax[blc_id][0].grid(b=True, which='both')
	ax[blc_id][0].plot(N[-print_last_tmstms:], T[-print_last_tmstms:, blc_id])
	ax[blc_id][0].set_ylabel(B_n_labels[blc_id], fontsize=fontsize)

	ax[blc_id][1].grid(b=True, which='both')
	ax[blc_id][1].plot(N[-print_last_tmstms:], kalmanT[-print_last_tmstms:, blc_id])

	ax[blc_id][2].grid(b=True, which='both')
	ax[blc_id][2].plot(N[-print_last_tmstms:], ma2T[-print_last_tmstms:, blc_id])

	ax[blc_id][3].grid(b=True, which='both')
	ax[blc_id][3].plot(N[-print_last_tmstms:], ma3T[-print_last_tmstms:, blc_id])

	ax[blc_id][4].grid(b=True, which='both')
	ax[blc_id][4].plot(N[-print_last_tmstms:], ma5T[-print_last_tmstms:, blc_id])

	ax[blc_id][5].grid(b=True, which='both')
	ax[blc_id][5].plot(N[-print_last_tmstms:], ma8T[-print_last_tmstms:, blc_id])

	ax[blc_id][6].grid(b=True, which='both')
	ax[blc_id][6].plot(N[-print_last_tmstms:], ma13T[-print_last_tmstms:, blc_id])

ax[0][0].set_title(r'RAW', fontsize=fontsize)
ax[0][1].set_title(r'Kalman filter', fontsize=fontsize)
ax[0][2].set_title(r'${MA}(2)$', fontsize=fontsize)
ax[0][3].set_title(r'${MA}(3)$', fontsize=fontsize)
ax[0][4].set_title(r'${MA}(5)$', fontsize=fontsize)
ax[0][5].set_title(r'${MA}(8)$', fontsize=fontsize)
ax[0][6].set_title(r'${MA}(13)$', fontsize=fontsize)
ax[9][0].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][1].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][2].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][3].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][4].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][5].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][6].set_xlabel(r'$n$', fontsize=fontsize)
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, currentfile+".png"))
plt.clf()
