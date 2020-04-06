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

currentfile = "data_T_1.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('kalmanT_dot', np.float_, (10,)),
                        ('ma3T', np.float_, (10,)),
                        ('ma13T', np.float_, (10,)),
                        ('ma55T', np.float_, (10,)),
                        ('ma144T', np.float_, (10,))])
(N, Mode, T,
    kalmanT, kalmanT_dot,
    ma3T, ma13T, ma55T, ma144T) = np.loadtxt(path.join(inpath, currentfile),
                                             unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

# print(np.amax(T), np.amin(T), np.mean(T))
# print(np.amax(kalmanT), np.amin(kalmanT), np.mean(kalmanT))
# print(np.amax(ma2T), np.amin(ma2T), np.mean(ma2T))
# print(np.amax(ma3T), np.amin(ma3T), np.mean(ma3T))
# print(np.amax(ma5T), np.amin(ma5T), np.mean(ma5T))
# print(np.amax(ma8T), np.amin(ma8T), np.mean(ma8T))
# print(np.amax(ma13T), np.amin(ma13T), np.mean(ma13T))
# print(np.amax(Mode), np.amin(Mode), np.mean(Mode))

B_n_labels = [r'$T^{(0)}$', r'$T^{(1)}$', r'$T^{(2)}$', r'$T^{(3)}$', r'$T^{(4)}$', r'$T^{(5)}$', r'$T^{(6)}$', r'$T^{(7)}$', r'$T^{(8)}$', r'$T^{(9)}$']
print_last_tmstms = 220
fontsize = 13
fig, ax = plt.subplots(nrows=10, ncols=7, constrained_layout=True, figsize=(15, 20))
for blc_id in range(0, 10):
    ax[blc_id][0].grid(b=True, which='both')
    ax[blc_id][0].plot(N[-print_last_tmstms:], T[-print_last_tmstms:, blc_id])
    ax[blc_id][0].set_ylabel(B_n_labels[blc_id], fontsize=fontsize)

    ax[blc_id][1].grid(b=True, which='both')
    ax[blc_id][1].plot(N[-print_last_tmstms:], kalmanT[-print_last_tmstms:, blc_id])

    ax[blc_id][2].grid(b=True, which='both')
    ax[blc_id][2].plot(N[-print_last_tmstms:], kalmanT_dot[-print_last_tmstms:, blc_id])

    ax[blc_id][3].grid(b=True, which='both')
    ax[blc_id][3].plot(N[-print_last_tmstms:], ma3T[-print_last_tmstms:, blc_id])

    ax[blc_id][4].grid(b=True, which='both')
    ax[blc_id][4].plot(N[-print_last_tmstms:], ma13T[-print_last_tmstms:, blc_id])

    ax[blc_id][5].grid(b=True, which='both')
    ax[blc_id][5].plot(N[-print_last_tmstms:], ma55T[-print_last_tmstms:, blc_id])

    ax[blc_id][6].grid(b=True, which='both')
    ax[blc_id][6].plot(N[-print_last_tmstms:], ma144T[-print_last_tmstms:, blc_id])

ax[0][0].set_title(r'RAW', fontsize=fontsize)
ax[0][1].set_title(r'$KF$ pos', fontsize=fontsize)
ax[0][2].set_title(r'$KF$ vel', fontsize=fontsize)
ax[0][3].set_title(r'${MA}(3)$', fontsize=fontsize)
ax[0][4].set_title(r'${MA}(13)$', fontsize=fontsize)
ax[0][5].set_title(r'${MA}(55)$', fontsize=fontsize)
ax[0][6].set_title(r'${MA}(144)$', fontsize=fontsize)
ax[9][0].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][1].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][2].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][3].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][4].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][5].set_xlabel(r'$n$', fontsize=fontsize)
ax[9][6].set_xlabel(r'$n$', fontsize=fontsize)
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, currentfile + ".png"))
plt.clf()
