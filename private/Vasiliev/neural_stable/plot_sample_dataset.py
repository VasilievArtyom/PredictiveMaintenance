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

currentfile = "data_T_664.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_, (5,)),
                        ('kalmanT', np.float_, (10,)),
                        ('kalmanT_dot', np.float_, (10,)),
                        ('rwavT', np.float_, (10,)),
                        ('ma13T', np.float_, (10,)),
                        ('ma55T', np.float_, (10,)),
                        ('ma144T', np.float_, (10,)),
                        ('S', np.float_, (10,)),
                        ('lfc', np.float_, (10,))])
(N, Mode,
    kalmanT, kalmanT_dot, rwavT,
    ma13T, ma55T, ma144T, S, lfc) = np.loadtxt(path.join(inpath, currentfile),
                                               unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

B_n_labels = [r'$T^{(0)}$', r'$T^{(1)}$', r'$T^{(2)}$', r'$T^{(3)}$', r'$T^{(4)}$', r'$T^{(5)}$', r'$T^{(6)}$', r'$T^{(7)}$', r'$T^{(8)}$', r'$T^{(9)}$']
print_last_tmstms = -0
fontsize = 13
blc_id = 0
fig, ax = plt.subplots(nrows=13, ncols=1, constrained_layout=True, figsize=(10, 15))

for r_index in range(0, 13):
    ax[r_index].grid(b=True, which='both')
    # ax[r_index].set_xlabel(r'$n$', fontsize=fontsize)
ax[0].set_title(B_n_labels[blc_id], fontsize=fontsize)
ax[0].plot(N[-print_last_tmstms:], kalmanT[-print_last_tmstms:, blc_id])
ax[0].set_ylabel(r'$KF$ pos', fontsize=fontsize)

ax[1].plot(N[-print_last_tmstms:], kalmanT_dot[-print_last_tmstms:, blc_id])
ax[1].set_ylabel(r'$KF$ vel', fontsize=fontsize)

ax[2].plot(N[-print_last_tmstms:], rwavT[-print_last_tmstms:, blc_id])
ax[2].set_ylabel(r'${\langle T \rangle}_w$', fontsize=fontsize)

ax[3].plot(N[-print_last_tmstms:], ma13T[-print_last_tmstms:, blc_id])
ax[3].set_ylabel(r'$MA(13)$', fontsize=fontsize)

ax[4].plot(N[-print_last_tmstms:], ma55T[-print_last_tmstms:, blc_id])
ax[4].set_ylabel(r'$MA(55)$', fontsize=fontsize)

ax[5].plot(N[-print_last_tmstms:], ma144T[-print_last_tmstms:, blc_id])
ax[5].set_ylabel(r'$MA(144)$', fontsize=fontsize)

ax[6].plot(N[-print_last_tmstms:], lfc[-print_last_tmstms:, blc_id])
ax[6].set_ylabel(r'$CfLF$', fontsize=fontsize)

ax[7].bar(N[-print_last_tmstms:], S[-print_last_tmstms:, blc_id])
ax[7].set_ylabel(r'$State$', fontsize=fontsize)

ax[8].bar(N[-print_last_tmstms:], Mode[-print_last_tmstms:, 0])
ax[8].set_ylabel(r'$Mode_0$', fontsize=fontsize)

ax[9].bar(N[-print_last_tmstms:], Mode[-print_last_tmstms:, 1])
ax[9].set_ylabel(r'$Mode_1$', fontsize=fontsize)

ax[10].bar(N[-print_last_tmstms:], Mode[-print_last_tmstms:, 2])
ax[10].set_ylabel(r'$Mode_2$', fontsize=fontsize)

ax[11].bar(N[-print_last_tmstms:], Mode[-print_last_tmstms:, 3])
ax[11].set_ylabel(r'$Mode_3$', fontsize=fontsize)

ax[12].bar(N[-print_last_tmstms:], Mode[-print_last_tmstms:, 4])
ax[12].set_ylabel(r'$Mode_4$', fontsize=fontsize)
ax[12].set_xlabel(r'$n$', fontsize=fontsize)

plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, currentfile + ".png"))
plt.clf()
