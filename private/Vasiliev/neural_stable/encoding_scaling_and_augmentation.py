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

currentfile = "RAWdataset.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.int_),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('kalmanT_dot', np.float_, (10,)),
                        ('rwavT', np.float_, (10,)),
                        ('ma13T', np.float_, (10,)),
                        ('ma55T', np.float_, (10,)),
                        ('ma144T', np.float_, (10,)),
                        ('S', np.int_, (10,)),
                        ('lfc', np.float_, (10,))])
(N, Mode, T,
    kalmanT, kalmanT_dot, rwavT,
    ma13T, ma55T, ma144T, S, lfc) = np.loadtxt(path.join(inpath, currentfile),
                                               unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


def scale(_data):
    _mean = np.mean(_data)
    _data = _data - _mean
    _len = np.amax(np.abs(_data))
    _data = _data * (1.0 / _len)
    return _data


def one_hot_encoder(_data):
    _category_vals = np.unique(_data)
    _encoded_data = np.zeros((len(_data), len(_category_vals)))
    for category_id, category_val in enumerate(_category_vals):
        for n in range(0, len(_data)):
            if (_data[n] == category_val):
                _encoded_data[n, category_id] = 1.0
    return _encoded_data


def print_dataset(_N, _Modes, _kalmanT, _kalmanT_dot, _rwavT, _ma13T, _ma55T, _ma144T, _S, _lfc, _index):
    outpath = "data/"
    currentfile = path.join(outpath, "data_T_{0}.csv".format(_index))
    f = open(currentfile, 'w+')
    print("N;Mode_0;Mode_1;Mode_2;Mode_3;Mode_4;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;kfT_dot_0;kfT_dot_1;kfT_dot_2;kfT_dot_3;kfT_dot_4;kfT_dot_5;kfT_dot_6;kfT_dot_7;kfT_dot_8;kfT_dot_9;rwavT_0;rwavT_1;rwavT_2;rwavT_3;rwavT_4;rwavT_5;rwavT_6;rwavT_7;rwavT_8;rwavT_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;ma55T_0;ma55T_1;ma55T_2;ma55T_3;ma55T_4;ma55T_5;ma55T_6;ma55T_7;ma55T_8;ma55T_9;ma144T_0;ma144T_1;ma144T_2;ma144T_3;ma144T_4;ma144T_5;ma144T_6;ma144T_7;ma144T_8;ma144T_9;S_0;S_1;S_2;S_3;S_4;S_5;S_6;S_7;S_8;S_9;lfc_0;lfc_1;lfc_2;lfc_3;lfc_4;lfc_5;lfc_6;lfc_7;lfc_8;lfc_9;", file=f)
    for line in range(0, len(_kalmanT[:, 0])):
        print(_N[line], _Modes[line, 0], _Modes[line, 1], _Modes[line, 2], _Modes[line, 3], _Modes[line, 4],
              _kalmanT[line, 0], _kalmanT[line, 1], _kalmanT[line, 2], _kalmanT[line, 3], _kalmanT[line, 4], _kalmanT[line, 5], _kalmanT[line, 6], _kalmanT[line, 7], _kalmanT[line, 8], _kalmanT[line, 9],
              _kalmanT_dot[line, 0], _kalmanT_dot[line, 1], _kalmanT_dot[line, 2], _kalmanT_dot[line, 3], _kalmanT_dot[line, 4], _kalmanT_dot[line, 5], _kalmanT_dot[line, 6], _kalmanT_dot[line, 7], _kalmanT_dot[line, 8], _kalmanT_dot[line, 9],
              _rwavT[line, 0], _rwavT[line, 1], _rwavT[line, 2], _rwavT[line, 3], _rwavT[line, 4], _rwavT[line, 5], _rwavT[line, 6], _rwavT[line, 7], _rwavT[line, 8], _rwavT[line, 9],
              _ma13T[line, 0], _ma13T[line, 1], _ma13T[line, 2], _ma13T[line, 3], _ma13T[line, 4], _ma13T[line, 5], _ma13T[line, 6], _ma13T[line, 7], _ma13T[line, 8], _ma13T[line, 9],
              _ma55T[line, 0], _ma55T[line, 1], _ma55T[line, 2], _ma55T[line, 3], _ma55T[line, 4], _ma55T[line, 5], _ma55T[line, 6], _ma55T[line, 7], _ma55T[line, 8], _ma55T[line, 9],
              _ma144T[line, 0], _ma144T[line, 1], _ma144T[line, 2], _ma144T[line, 3], _ma144T[line, 4], _ma144T[line, 5], _ma144T[line, 6], _ma144T[line, 7], _ma144T[line, 8], _ma144T[line, 9],
              _S[line, 0], _S[line, 1], _S[line, 2], _S[line, 3], _S[line, 4], _S[line, 5], _S[line, 6], _S[line, 7], _S[line, 8], _S[line, 9],
              _lfc[line, 0], _lfc[line, 1], _lfc[line, 2], _lfc[line, 3], _lfc[line, 4], _lfc[line, 5], _lfc[line, 6], _lfc[line, 7], _lfc[line, 8], _lfc[line, 9],
              sep=';', file=f)


Modes = one_hot_encoder(Mode)

skalmanT = np.empty(shape(kalmanT))
skalmanT_dot = np.empty(shape(kalmanT_dot))
srwavT = np.empty(shape(rwavT))
sma13T = np.empty(shape(ma13T))
sma55T = np.empty(shape(ma55T))
sma144T = np.empty(shape(ma144T))
slfc = np.empty(shape(lfc))
skalmanT.fill(np.nan)
skalmanT_dot.fill(np.nan)
srwavT.fill(np.nan)
sma13T.fill(np.nan)
sma55T.fill(np.nan)
sma144T.fill(np.nan)
slfc.fill(np.nan)
frst_fail = [75, 75, 75, 78, 38, 74, 0, 38, 38, 14]
for blc_id in range(0, 10):
    skalmanT[:, blc_id] = scale(kalmanT[:, blc_id])
    skalmanT_dot[:, blc_id] = scale(kalmanT_dot[:, blc_id])
    srwavT[:, blc_id] = scale(rwavT[:, blc_id])
    sma13T[13:, blc_id] = scale(ma13T[13:, blc_id])
    sma55T[55:, blc_id] = scale(ma55T[55:, blc_id])
    sma144T[144:, blc_id] = scale(ma144T[144:, blc_id])
    slfc[frst_fail[blc_id]:, blc_id] = scale(lfc[frst_fail[blc_id]:, blc_id])

print_dataset(N, Modes, skalmanT, skalmanT_dot, srwavT, sma13T, sma55T, sma144T, S.astype('float'), slfc, 0)

np.random.seed(0)
agmntCount = 1000

for agmnt_index in range(1, agmntCount + 1):
    agmnt_kalmanT = np.empty(shape(kalmanT))
    agmnt_kalmanT_dot = np.empty(shape(kalmanT_dot))
    agmnt_rwavT = np.empty(shape(rwavT))
    agmnt_ma13T = np.empty(shape(ma13T))
    agmnt_ma55T = np.empty(shape(ma55T))
    agmnt_ma144T = np.empty(shape(ma144T))
    agmnt_kalmanT.fill(np.nan)
    agmnt_kalmanT_dot.fill(np.nan)
    agmnt_rwavT.fill(np.nan)
    agmnt_ma13T.fill(np.nan)
    agmnt_ma55T.fill(np.nan)
    agmnt_ma144T.fill(np.nan)

    for blc_id in range(0, 10):
        mu = 0
        sigma = np.mean(np.abs(skalmanT[:-2, :] - skalmanT[1:-1, :])) / 2.0
        agmnt_kalmanT[:, blc_id] = skalmanT[:, blc_id] + np.random.normal(mu, sigma, (np.size(skalmanT[:, 0])))

        sigma = np.mean(np.abs(skalmanT_dot[:-2, :] - skalmanT_dot[1:-1, :])) / 10.0
        agmnt_kalmanT_dot[:, blc_id] = skalmanT_dot[:, blc_id] + np.random.normal(mu, sigma, (np.size(skalmanT_dot[:, 0])))

        sigma = np.mean(np.abs(srwavT[:-2, :] - srwavT[1:-1, :])) / 2.0
        agmnt_rwavT[:, blc_id] = srwavT[:, blc_id] + np.random.normal(mu, sigma, (np.size(srwavT[:, 0])))

        sigma = np.mean(np.abs(sma13T[13:-2, :] - sma13T[14:-1, :])) / 10.0
        agmnt_ma13T[:, blc_id] = sma13T[:, blc_id] + np.random.normal(mu, sigma, (np.size(sma13T[:, 0])))

        sigma = np.mean(np.abs(sma55T[55:-2, :] - sma13T[56:-1, :])) / 2.0
        agmnt_ma55T[:, blc_id] = sma55T[:, blc_id] + np.random.normal(mu, sigma, (np.size(sma55T[:, 0])))

        sigma = np.mean(np.abs(sma13T[144:-2, :] - sma13T[145:-1, :])) / 2.0
        agmnt_ma144T[:, blc_id] = sma144T[:, blc_id] + np.random.normal(mu, sigma, (np.size(sma144T[:, 0])))
    print_dataset(N, Modes, agmnt_kalmanT, agmnt_kalmanT_dot, agmnt_rwavT, agmnt_ma13T, agmnt_ma55T, agmnt_ma144T, S.astype('float'), slfc, agmnt_index)
