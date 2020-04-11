import math
import numpy as np
from numpy import *
from os import path
import os
import sys
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
np.random.seed(0)  # Set a random seed for reproducibility

# <--------------------->
# Tunable

rnn_sequence_length = 300
cutFromTail = 1
cutFromHead = 144
max_pred_step = 60
# <--------------------->

tmstmsp = 2353
if len(sys.argv) > 1:
    tmstmsp = int(sys.argv[1])


def read_dtaset_by_index(index):
    inpath = "../data/"
    currentfile = path.join(inpath, "data_T_{0}.csv".format(index))
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
    # N, Mode, kalmanT, kalmanT_dot, rwavT, ma13T, ma55T, ma144T, S, lfc
    return np.loadtxt(currentfile, unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


# Read unaugmented dataset
N, Mode, kalmanT, kalmanT_dot, rwavT, ma13T, ma55T, ma144T, S, lfc = read_dtaset_by_index(0)
# Collect full dataset
n_features = 13
l_b, r_b = cutFromHead, cutFromTail
N = N[l_b:-r_b]
ds = np.empty((10, len(N), n_features))
for _blc_id in range(0, 10):
    N, Mode, kalmanT, kalmanT_dot, rwavT, ma13T, ma55T, ma144T, S, lfc = read_dtaset_by_index(0)
    (ds[_blc_id, :, 0], ds[_blc_id, :, 1], ds[_blc_id, :, 2],
     ds[_blc_id, :, 3], ds[_blc_id, :, 4], ds[_blc_id, :, 5],
     ds[_blc_id, :, 6], ds[_blc_id, :, 7], ds[_blc_id, :, 8:13]) = (kalmanT[l_b:-r_b, _blc_id], kalmanT_dot[l_b:-r_b, _blc_id],
                                                                    rwavT[l_b:-r_b, _blc_id], ma13T[l_b:-r_b, _blc_id], ma55T[l_b:-r_b, _blc_id],
                                                                    ma144T[l_b:-r_b, _blc_id], S[l_b:-r_b, _blc_id], lfc[l_b:-r_b, _blc_id], Mode[l_b:-r_b, :])
S = S[l_b:-r_b, :]


def read_singlemodelpredictions(_blc_id, _pred_step):
    inpath = ""
    currentfile = str(_blc_id) + '_binary_on_' + str(_pred_step) + ".csv"
    # Read from file
    strdatatype = np.dtype([('N', np.int_), ('GT', np.float_), ('pred', np.float_)])
    # _N, _GT, _pred
    return np.loadtxt(path.join(inpath, currentfile), unpack=True, delimiter=',', skiprows=1, dtype=strdatatype)


def collect_predictions_over_tmstmp(_n):
    preds = np.zeros((21, 10))
    for blc_id in range(0, 10):
        for pred_step in range(1, 22):
            _N, _GT, _pred = read_singlemodelpredictions(blc_id, pred_step)
            min_poss_n, max_poss_n = np.amin(_N), np.amax(_N)
            assert ((_n <= max_poss_n) and (_n >= min_poss_n)), "Out of bounds"
            index = int(np.where(_N == _n)[0])
            preds[pred_step - 1, blc_id] = _pred[index]
    return preds


outpath = "../results/"
predictions = collect_predictions_over_tmstmp(tmstmsp)
S_n_labels = [r'$S^{(0)}$', r'$S^{(1)}$', r'$S^{(2)}$', r'$S^{(3)}$', r'$B^{(4)}$', r'$S^{(5)}$', r'$S^{(6)}$', r'$S^{(7)}$', r'$S^{(8)}$', r'$S^{(9)}$']
fig, ax = plt.subplots(nrows=10, ncols=1, constrained_layout=True, figsize=(10, 13))
for rows in range(0, 10):
    ax[rows].step(N[-200:-60], S[-200:-60, rows], label="Raw signal")
    ax[rows].step(N[-60:], S[-60:, rows], label="Validation", color='gray')
    ax[rows].plot(np.arange(tmstmsp, tmstmsp + 21), predictions[:, rows], label="Prediction")
    ax[rows].set_ylabel(S_n_labels[rows], fontsize=15)
    ax[rows].legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, str(tmstmsp) + ".png"))
plt.clf()
