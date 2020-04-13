import math
import numpy as np
from numpy import *
from os import path
import os
import sys
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
np.random.seed(0)  # Set a random seed for reproducibility

blc_id = 0
pred_step = 1
if len(sys.argv) > 1:
    blc_id = int(sys.argv[1])
    pred_step = int(sys.argv[2])

# <--------------------->
# Tunable

rnn_sequence_length = 300
cutFromTail = 60
cutFromHead = 144
max_pred_step = 60
# <--------------------->


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
    (ds[_blc_id, :, 0], ds[_blc_id, :, 1], ds[_blc_id, :, 2],
     ds[_blc_id, :, 3], ds[_blc_id, :, 4], ds[_blc_id, :, 5],
     ds[_blc_id, :, 6], ds[_blc_id, :, 7], ds[_blc_id, :, 8:13]) = (kalmanT[l_b:-r_b, _blc_id], kalmanT_dot[l_b:-r_b, _blc_id],
                                                                    rwavT[l_b:-r_b, _blc_id], ma13T[l_b:-r_b, _blc_id], ma55T[l_b:-r_b, _blc_id],
                                                                    ma144T[l_b:-r_b, _blc_id], S[l_b:-r_b, _blc_id], lfc[l_b:-r_b, _blc_id], Mode[l_b:-r_b, :])
S = S[l_b:-r_b, :]
min_tmpstmp = rnn_sequence_length + 144
max_tmstmp = len(N) - max_pred_step
B_n_labels = [r'$B^{(0)}$', r'$B^{(1)}$', r'$B^{(2)}$', r'$B^{(3)}$', r'$B^{(4)}$', r'$B^{(5)}$', r'$B^{(6)}$', r'$B^{(7)}$', r'$B^{(8)}$', r'$B^{(9)}$']
currentfile = str(blc_id) + '_binary_on_' + str(pred_step) + ".csv"
inpath = ""
outpath = "plots"
# Read from file
strdatatype = np.dtype([('N', np.int_), ('GT', np.float_), ('pred', np.float_)])
N, GT, pred = np.loadtxt(path.join(inpath, currentfile),
                         unpack=True, delimiter=',', skiprows=1, dtype=strdatatype)

print(blc_id, pred_step)


def count_mismatches(_level):
    current_count = 0
    for tmstmp in range(0, len(GT)):
        true_val = GT[tmstmp]
        pred_val = pred[tmstmp]
        tmpval = 0.0
        if (pred_val > _level):
            tmpval = 1.0
        if (int(tmpval) != true_val):
            current_count += 1
    return current_count


error = np.inf
level = np.inf
for tmp_level in np.arange(0, 1.0, 0.0001):
    curr_error = count_mismatches(tmp_level)
    if (curr_error < error):
        error = curr_error
        level = tmp_level
print(error, level)
f = open(('thresholds/thresholds_for_blc_' + str(blc_id) + ".txt"), 'a+')
print(level, file=f)

# print_last_tmstms = -0
# fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 1.8))
# ax.set_title("Prediction on " + str(pred_step) + " steps for " + B_n_labels[blc_id])
# ax.step(N[-print_last_tmstms:], GT[-print_last_tmstms:], label="GT")
# ax.plot(N[-print_last_tmstms:], pred[-print_last_tmstms:], label="Prediction")

# plt.tight_layout()
# ax.legend(loc='best', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, str(blc_id) + '_binary_on_' + str(pred_step) + ".png"))
# plt.clf()
