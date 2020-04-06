import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import os
import matplotlib.pyplot as plt
from pykalman import KalmanFilter


plt.rc('text', usetex=True)

outpath = ""
inpath = "../../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_), ('Mode', np.int_),
                        ('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_)])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

r = np.array([[1., 0.64405472, 0.50105597, 0.59813306, 0.5884463, 0.58708052, 0.52039093, 0.54867688, 0.5868079, 0.53204574],
              [0.64405472, 1., 0.54280639, 0.56255202, 0.43551609, 0.50994087, 0.60364561, 0.44044939, 0.60626518, 0.5784179],
              [0.50105597, 0.54280639, 1., 0.50643918, 0.47961374, 0.56160455, 0.54269385, 0.34935579, 0.64008289, 0.62772758],
              [0.59813306, 0.56255202, 0.50643918, 1., 0.53705638, 0.63280362, 0.59218442, 0.39130634, 0.5421819, 0.55723074],
              [0.5884463, 0.43551609, 0.47961374, 0.53705638, 1., 0.63135103, 0.61660445, 0.63813615, 0.54793749, 0.60623637],
              [0.58708052, 0.50994087, 0.56160455, 0.63280362, 0.63135103, 1., 0.61683971, 0.47048264, 0.58033179, 0.68790507],
              [0.52039093, 0.60364561, 0.54269385, 0.59218442, 0.61660445, 0.61683971, 1., 0.41857105, 0.66912111, 0.72242696],
              [0.54867688, 0.44044939, 0.34935579, 0.39130634, 0.63813615, 0.47048264, 0.41857105, 1., 0.44207233, 0.43667893],
              [0.5868079, 0.60626518, 0.64008289, 0.5421819, 0.54793749, 0.58033179, 0.66912111, 0.44207233, 1., 0.70436581],
              [0.53204574, 0.5784179, 0.62772758, 0.55723074, 0.60623637, 0.68790507, 0.72242696, 0.43667893, 0.70436581, 1.]])
print(shape(r))

# kalman filter
kalmanT = np.zeros((len(T[:, 0]), 10))
kalmanT_dot = np.zeros((len(T[:, 0]), 10))
for blc_id in range(0, 10):
    kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
                      transition_covariance=0.01 * np.eye(2))
    states_pred = kf.em(T[:, blc_id]).smooth(T[:, blc_id])[0]
    kalmanT[:, blc_id] = states_pred[:, 0]
    kalmanT_dot[:, blc_id] = states_pred[:, 1]

# r-weighted average
rwavT = np.zeros((len(T[:, 0]), 10))
for blc_id_i in range(0, 10):
    for blc_id_j in range(0, 10):
        if (blc_id_i != blc_id_j):
            rwavT[:, blc_id_i] += r[blc_id_i, blc_id_j] * kalmanT[:, blc_id_j]
    rwavT[:, blc_id_i] /= 9.0

# moving average
ma_lags = [13, 55, 144]
maT = np.zeros((len(ma_lags), len(T[:, 0]), 10))
for i, ma_lag in enumerate(ma_lags):
    maT[i, 0:ma_lag, :].fill(np.nan)
    for j in range(ma_lag, len(T[:, 0])):
        maT[i, j, :] = np.mean(kalmanT[j - ma_lag:j, :], axis=0)

# counts from last failure
lfc = np.empty((len(T[:, 0]), 10))
lfc.fill(np.nan)
for blc_id in range(0, 10):
    # search for first failure index
    tmp_pointer = 0
    while (S[tmp_pointer, blc_id]):
        tmp_pointer += 1
    for n in range(tmp_pointer, len(T[:, 0])):
        failure_pointer = 0
        while (S[n - failure_pointer, blc_id]):
            failure_pointer += 1
        lfc[n, blc_id] = failure_pointer
# print(np.unique(lfc[75:, 0], return_counts=True))
# print(np.unique(lfc[75:, 1], return_counts=True))
# print(np.unique(lfc[75:, 2], return_counts=True))
# print(np.unique(lfc[78:, 3], return_counts=True))
# print(np.unique(lfc[38:, 4], return_counts=True))
# print(np.unique(lfc[74:, 5], return_counts=True))
# print(np.unique(lfc[0:, 6], return_counts=True))
# print(np.unique(lfc[38:, 7], return_counts=True))
# print(np.unique(lfc[38:, 8], return_counts=True))
# print(np.unique(lfc[14:, 9], return_counts=True))

f = open('data/RAWdataset.csv', 'w+')
print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;kfT_dot_0;kfT_dot_1;kfT_dot_2;kfT_dot_3;kfT_dot_4;kfT_dot_5;kfT_dot_6;kfT_dot_7;kfT_dot_8;kfT_dot_9;rwavT_0;rwavT_1;rwavT_2;rwavT_3;rwavT_4;rwavT_5;rwavT_6;rwavT_7;rwavT_8;rwavT_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;ma55T_0;ma55T_1;ma55T_2;ma55T_3;ma55T_4;ma55T_5;ma55T_6;ma55T_7;ma55T_8;ma55T_9;ma144T_0;ma144T_1;ma144T_2;ma144T_3;ma144T_4;ma144T_5;ma144T_6;ma144T_7;ma144T_8;ma144T_9;S_0;S_1;S_2;S_3;S_4;S_5;S_6;S_7;S_8;S_9;lfc_0;lfc_1;lfc_2;lfc_3;lfc_4;lfc_5;lfc_6;lfc_7;lfc_8;lfc_9;", file=f)
for line in range(0, len(T[:, 0])):
    print(N[line, 0], Mode[line],
          T[line, 0], T[line, 1], T[line, 2], T[line, 3], T[line, 4], T[line, 5], T[line, 6], T[line, 7], T[line, 8], T[line, 9],
          kalmanT[line, 0], kalmanT[line, 1], kalmanT[line, 2], kalmanT[line, 3], kalmanT[line, 4], kalmanT[line, 5], kalmanT[line, 6], kalmanT[line, 7], kalmanT[line, 8], kalmanT[line, 9],
          kalmanT_dot[line, 0], kalmanT_dot[line, 1], kalmanT_dot[line, 2], kalmanT_dot[line, 3], kalmanT_dot[line, 4], kalmanT_dot[line, 5], kalmanT_dot[line, 6], kalmanT_dot[line, 7], kalmanT_dot[line, 8], kalmanT_dot[line, 9],
          rwavT[line, 0], rwavT[line, 1], rwavT[line, 2], rwavT[line, 3], rwavT[line, 4], rwavT[line, 5], rwavT[line, 6], rwavT[line, 7], rwavT[line, 8], rwavT[line, 9],
          maT[0, line, 0], maT[0, line, 1], maT[0, line, 2], maT[0, line, 3], maT[0, line, 4], maT[0, line, 5], maT[0, line, 6], maT[0, line, 7], maT[0, line, 8], maT[0, line, 9],
          maT[1, line, 0], maT[1, line, 1], maT[1, line, 2], maT[1, line, 3], maT[1, line, 4], maT[1, line, 5], maT[1, line, 6], maT[1, line, 7], maT[1, line, 8], maT[1, line, 9],
          maT[2, line, 0], maT[2, line, 1], maT[2, line, 2], maT[2, line, 3], maT[2, line, 4], maT[2, line, 5], maT[2, line, 6], maT[2, line, 7], maT[2, line, 8], maT[2, line, 9],
          S[line, 0].astype('int'), S[line, 1].astype('int'), S[line, 2].astype('int'), S[line, 3].astype('int'), S[line, 4].astype('int'), S[line, 5].astype('int'), S[line, 6].astype('int'), S[line, 7].astype('int'), S[line, 8].astype('int'), S[line, 9].astype('int'),
          lfc[line, 0], lfc[line, 1], lfc[line, 2], lfc[line, 3], lfc[line, 4], lfc[line, 5], lfc[line, 6], lfc[line, 7], lfc[line, 8], lfc[line, 9],
          sep=';', file=f)
