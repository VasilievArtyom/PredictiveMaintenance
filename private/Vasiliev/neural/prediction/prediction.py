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

thresholds = np.array([[0.7363000000000001, 0.38370000000000004, 0.515, 0.4349, 0.3808, 0.4183, 0.4733, 0.5197, 0.5189, 0.4279, 0.5872, 0.4781, 0.5097, 0.4874, 0.5602, 0.6259, 0.6271, 0.5283, 0.5773, 0.5309, 0.514],
                       [0.44120000000000004, 0.4197, 0.3784, 0.47390000000000004, 0.5248, 0.5379, 0.7076, 0.5175000000000001, 0.5910000000000001, 0.4786, 0.5700000000000001, 0.4707, 0.5482, 0.5637, 0.6167, 0.4721, 0.6083000000000001, 0.6224000000000001, 0.5303, 0.3889, 0.5816],
                       [0.672, 0.5526, 0.3886, 0.5494, 0.6135, 0.4335, 0.5271, 0.4697, 0.3855, 0.36210000000000003, 0.4665, 0.4539, 0.4222, 0.47340000000000004, 0.6235, 0.6546000000000001, 0.5111, 0.41490000000000005, 0.537, 0.49660000000000004, 0.4545],
                       [0.6036, 0.32130000000000003, 0.39730000000000004, 0.5054000000000001, 0.3912, 0.5115000000000001, 0.5966, 0.4303, 0.4874, 0.5581, 0.5870000000000001, 0.44020000000000004, 0.5082, 0.5076, 0.5948, 0.5859, 0.6184000000000001, 0.4621, 0.5294, 0.49870000000000003, 0.5145000000000001],
                       [0.4832, 0.6228, 0.48460000000000003, 0.3685, 0.4339, 0.449, 0.38, 0.4142, 0.3724, 0.40190000000000003, 0.45790000000000003, 0.47440000000000004, 0.5237, 0.5003000000000001, 0.5287000000000001, 0.5952000000000001, 0.4151, 0.41800000000000004, 0.49250000000000005, 0.5182, 0.4697],
                       [0.2577, 0.3844, 0.4595, 0.47090000000000004, 0.2, 0.4587, 0.46390000000000003, 0.5737, 0.4501, 0.41250000000000003, 0.5215000000000001, 0.4838, 0.4988, 0.5387000000000001, 0.4529, 0.5642, 0.6025, 0.4297, 0.47050000000000003, 0.5363, 0.40940000000000004],
                       [0.5703, 0.4036, 0.4333, 0.4232, 0.3195, 0.42900000000000005, 0.3451, 0.3924, 0.41240000000000004, 0.5109, 0.5128, 0.4612, 0.5228, 0.5812, 0.4897, 0.6432, 0.6041000000000001, 0.3427, 0.32370000000000004, 0.5186000000000001, 0.2807],
                       [0.6078, 0.4555, 0.4121, 0.5043, 0.2179, 0.3653, 0.42960000000000004, 0.46840000000000004, 0.41240000000000004, 0.4012, 0.39630000000000004, 0.5145000000000001, 0.4707, 0.3925, 0.4984, 0.4641, 0.5858, 0.46990000000000004, 0.4373, 0.48610000000000003, 0.4218],
                       [0.6316, 0.4893, 0.3133, 0.3491, 0.2646, 0.40950000000000003, 0.2841, 0.2679, 0.5311, 0.43370000000000003, 0.5479, 0.5695, 0.4565, 0.4097, 0.5334, 0.5807, 0.5993, 0.504, 0.5444, 0.0, 0.45790000000000003],
                       [0.39590000000000003, 0.32, 0.4315, 0.3291, 0.4738, 0.45030000000000003, 0.37, 0.4106, 0.4877, 0.626, 0.616, 0.4469, 0.48310000000000003, 0.5291, 0.5963, 0.5992000000000001, 0.6589, 0.534, 0.5568000000000001, 0.4786, 0.5087]])

print(shape(thresholds))


def read_singlemodelpredictions(_blc_id, _pred_step):
    inpath = ""
    currentfile = str(_blc_id) + '_binary_on_' + str(_pred_step) + ".csv"
    # Read from file
    strdatatype = np.dtype([('N', np.int_), ('GT', np.float_), ('pred', np.float_)])
    # _N, _GT, _pred
    return np.loadtxt(path.join(inpath, currentfile), unpack=True, delimiter=',', skiprows=1, dtype=strdatatype)


def collect_predictions_over_tmstmp(_n):
    preds = np.zeros((21, 10))
    th_preds = np.zeros((21, 10))
    for blc_id in range(0, 10):
        for pred_step in range(1, 22):
            _N, _GT, _pred = read_singlemodelpredictions(blc_id, pred_step)
            min_poss_n, max_poss_n = np.amin(_N), np.amax(_N)
            assert ((_n <= max_poss_n) and (_n >= min_poss_n)), "Out of bounds"
            index = int(np.where(_N == _n)[0])
            preds[pred_step - 1, blc_id] = _pred[index]
            if (_pred[index] >= thresholds[blc_id, pred_step - 1]):
                th_preds[pred_step - 1, blc_id] = 1.0
    return preds, th_preds


outpath = "../results/"
predictions, th_predictions = collect_predictions_over_tmstmp(tmstmsp)
S_n_labels = [r'$S^{(0)}$', r'$S^{(1)}$', r'$S^{(2)}$', r'$S^{(3)}$', r'$B^{(4)}$', r'$S^{(5)}$', r'$S^{(6)}$', r'$S^{(7)}$', r'$S^{(8)}$', r'$S^{(9)}$']
fig, ax = plt.subplots(nrows=10, ncols=1, constrained_layout=True, figsize=(11, 15))
for rows in range(0, 10):
    ax[rows].step(N[-100:-60], S[-100:-60, rows], label="Raw signal")
    ax[rows].step(N[-60:], S[-60:, rows], label="Validation", color='gray')
    ax[rows].plot(np.arange(tmstmsp, tmstmsp + 21), predictions[:, rows], label="Prediction value", alpha=0.5, color='red')
    ax[rows].step(np.arange(tmstmsp, tmstmsp + 21), th_predictions[:, rows], label="Prediction thresholded", color='green', linewidth=3, alpha=0.5)
    ax[rows].set_ylabel(S_n_labels[rows], fontsize=15)
    ax[rows].legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, str(tmstmsp) + ".png"))
plt.clf()
