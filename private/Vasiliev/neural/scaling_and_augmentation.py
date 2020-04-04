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

currentfile = "RAWdataset.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.int_),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('ma2T', np.float_, (10,)),
                        ('ma3T', np.float_, (10,)),
                        ('ma5T', np.float_, (10,)),
                        ('ma8T', np.float_, (10,)),
                        ('ma13T', np.float_, (10,)),
                        ('ma21T', np.float_, (10,)),
                        ('ma34T', np.float_, (10,)),
                        ('ma55T', np.float_, (10,)),
                        ('ma89T', np.float_, (10,)),
                        ('ma144T', np.float_, (10,))])
N, Mode, T, kalmanT, ma2T, ma3T, ma5T, ma8T, ma13T, ma21T, ma34T, ma55T, ma89T, ma144T = np.loadtxt(path.join(inpath, currentfile),
                                                                                                    unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
# map from
print(np.amax(T), np.amin(T), np.mean(T))
print(np.amax(kalmanT), np.amin(kalmanT), np.mean(kalmanT))
print(np.amax(ma2T), np.amin(ma2T), np.mean(ma2T))
print(np.amax(ma3T), np.amin(ma3T), np.mean(ma3T))
print(np.amax(ma5T), np.amin(ma5T), np.mean(ma5T))
print(np.amax(ma8T), np.amin(ma8T), np.mean(ma8T))
print(np.amax(ma13T), np.amin(ma13T), np.mean(ma13T))
print(np.amax(ma21T), np.amin(ma21T), np.mean(ma21T))
print(np.amax(ma34T), np.amin(ma34T), np.mean(ma34T))
print(np.amax(ma55T), np.amin(ma55T), np.mean(ma55T))
print(np.amax(ma89T), np.amin(ma89T), np.mean(ma89T))
print(np.amax(ma144T), np.amin(ma144T), np.mean(ma144T))
print(np.amax(Mode), np.amin(Mode), np.mean(Mode))

# map to [-1, 1] with zero mean


def scale_sinle_series(_data, _data_maxval, _data_minval, _data_mean, _fill_latency):
    up_len = np.abs(np.abs(_data_maxval) - np.abs(_data_mean))
    dn_len = np.abs(np.abs(_data_minval) - np.abs(_data_mean))
    tmp_data = np.array(_data - _data_mean)
    for i in range(0, len(_data)):
        if(tmp_data[i] > 0.0):
            tmp_data[i] = np.divide(tmp_data[i], up_len) * _fill_latency
        else:
            tmp_data[i] = np.divide(tmp_data[i], dn_len) * _fill_latency
    return tmp_data


sT = np.zeros((len(T[:, 0]), 10))
skalmanT = np.zeros((len(T[:, 0]), 10))
sma2T = np.zeros((len(T[:, 0]), 10))
sma3T = np.zeros((len(T[:, 0]), 10))
sma5T = np.zeros((len(T[:, 0]), 10))
sma8T = np.zeros((len(T[:, 0]), 10))
sma13T = np.zeros((len(T[:, 0]), 10))
sma21T = np.zeros((len(T[:, 0]), 10))
sma34T = np.zeros((len(T[:, 0]), 10))
sma55T = np.zeros((len(T[:, 0]), 10))
sma89T = np.zeros((len(T[:, 0]), 10))
sma144T = np.zeros((len(T[:, 0]), 10))
sma3T[:3, :].fill(np.nan)
sma2T[:2, :].fill(np.nan)
sma5T[:5, :].fill(np.nan)
sma8T[:8, :].fill(np.nan)
sma13T[:13, :].fill(np.nan)
sma21T[:21, :].fill(np.nan)
sma34T[:34, :].fill(np.nan)
sma55T[:55, :].fill(np.nan)
sma89T[:89, :].fill(np.nan)
sma144T[:144, :].fill(np.nan)


fill_latency_for_T = 0.90
for blc_id in range(0, 10):
    sT[:, blc_id] = scale_sinle_series(T[:, blc_id], np.amax(T[:, blc_id]), np.amin(T[:, blc_id]), np.mean(T[:, blc_id]), fill_latency_for_T)
    skalmanT[:, blc_id] = scale_sinle_series(kalmanT[:, blc_id], np.amax(kalmanT[:, blc_id]), np.amin(kalmanT[:, blc_id]), np.mean(kalmanT[:, blc_id]), fill_latency_for_T)
    sma2T[2:, blc_id] = scale_sinle_series(ma2T[2:, blc_id], np.amax(ma2T[2:, blc_id]), np.amin(ma2T[2:, blc_id]), np.mean(ma2T[2:, blc_id]), fill_latency_for_T)
    sma3T[3:, blc_id] = scale_sinle_series(ma3T[3:, blc_id], np.amax(ma3T[3:, blc_id]), np.amin(ma3T[3:, blc_id]), np.mean(ma3T[3:, blc_id]), fill_latency_for_T)
    sma5T[5:, blc_id] = scale_sinle_series(ma5T[5:, blc_id], np.amax(ma5T[5:, blc_id]), np.amin(ma5T[5:, blc_id]), np.mean(ma5T[5:, blc_id]), fill_latency_for_T)
    sma8T[8:, blc_id] = scale_sinle_series(ma8T[8:, blc_id], np.amax(ma8T[8:, blc_id]), np.amin(ma8T[8:, blc_id]), np.mean(ma8T[8:, blc_id]), fill_latency_for_T)
    sma13T[13:, blc_id] = scale_sinle_series(ma13T[13:, blc_id], np.amax(ma13T[13:, blc_id]), np.amin(ma13T[13:, blc_id]), np.mean(ma13T[13:, blc_id]), fill_latency_for_T)
    sma21T[21:, blc_id] = scale_sinle_series(ma21T[21:, blc_id], np.amax(ma21T[21:, blc_id]), np.amin(ma21T[21:, blc_id]), np.mean(ma21T[21:, blc_id]), fill_latency_for_T)
    sma34T[34:, blc_id] = scale_sinle_series(ma34T[34:, blc_id], np.amax(ma34T[34:, blc_id]), np.amin(ma34T[34:, blc_id]), np.mean(ma34T[34:, blc_id]), fill_latency_for_T)
    sma55T[55:, blc_id] = scale_sinle_series(ma55T[55:, blc_id], np.amax(ma55T[55:, blc_id]), np.amin(ma55T[55:, blc_id]), np.mean(ma55T[55:, blc_id]), fill_latency_for_T)
    sma89T[89:, blc_id] = scale_sinle_series(ma89T[89:, blc_id], np.amax(ma89T[89:, blc_id]), np.amin(ma89T[89:, blc_id]), np.mean(ma89T[89:, blc_id]), fill_latency_for_T)
    sma144T[144:, blc_id] = scale_sinle_series(ma144T[144:, blc_id], np.amax(ma144T[144:, blc_id]), np.amin(ma144T[144:, blc_id]), np.mean(ma144T[144:, blc_id]), fill_latency_for_T)

sMode = scale_sinle_series(Mode, np.amax(Mode), np.amin(Mode), np.mean(Mode), 1.0)
print("<--------------------------------->")
print(np.amax(sT), np.amin(sT), np.mean(sT))
print(np.amax(skalmanT), np.amin(skalmanT), np.mean(skalmanT))
print(np.amax(sma2T), np.amin(sma2T), np.mean(sma2T))
print(np.amax(sma3T), np.amin(sma3T), np.mean(sma3T))
print(np.amax(sma5T), np.amin(sma5T), np.mean(sma5T))
print(np.amax(sma8T), np.amin(sma8T), np.mean(sma8T))
print(np.amax(sma13T), np.amin(sma13T), np.mean(sma13T))
print(np.amax(sma21T), np.amin(sma21T), np.mean(sma21T))
print(np.amax(sma34T), np.amin(sma34T), np.mean(sma34T))
print(np.amax(sma55T), np.amin(sma55T), np.mean(sma55T))
print(np.amax(sma89T), np.amin(sma89T), np.mean(sma89T))
print(np.amax(sma144T), np.amin(sma144T), np.mean(sma144T))
print(np.amax(sMode), np.amin(sMode), np.mean(sMode))

f = open('data/data_T_0.csv', 'w+')
print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;ma2T_0;ma2T_1;ma2T_2;ma2T_3;ma2T_4;ma2T_5;ma2T_6;ma2T_7;ma2T_8;ma2T_9;ma3T_0;ma3T_1;ma3T_2;ma3T_3;ma3T_4;ma3T_5;ma3T_6;ma3T_7;ma3T_8;ma3T_9;ma5T_0;ma5T_1;ma5T_2;ma5T_3;ma5T_4;ma5T_5;ma5T_6;ma5T_7;ma5T_8;ma5T_9;ma8T_0;ma8T_1;ma8T_2;ma8T_3;ma8T_4;ma8T_5;ma8T_6;ma8T_7;ma8T_8;ma8T_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;ma21T_0;ma21T_1;ma21T_2;ma21T_3;ma21T_4;ma21T_5;ma21T_6;ma21T_7;ma21T_8;ma21T_9;ma34T_0;ma34T_1;ma34T_2;ma34T_3;ma34T_4;ma34T_5;ma34T_6;ma34T_7;ma34T_8;ma34T_9;ma55T_0;ma55T_1;ma55T_2;ma55T_3;ma55T_4;ma55T_5;ma55T_6;ma55T_7;ma55T_8;ma55T_9;ma89T_0;ma89T_1;ma89T_2;ma89T_3;ma89T_4;ma89T_5;ma89T_6;ma89T_7;ma89T_8;ma89T_9;ma144T_0;ma144T_1;ma144T_2;ma144T_3;ma144T_4;ma144T_5;ma144T_6;ma144T_7;ma144T_8;ma144T_9;", file=f)
for line in range(0, len(T[:, 0])):
    print(N[line], sMode[line],
          sT[line, 0], sT[line, 1], sT[line, 2], sT[line, 3], sT[line, 4], sT[line, 5], sT[line, 6], sT[line, 7], sT[line, 8], sT[line, 9],
          skalmanT[line, 0], skalmanT[line, 1], skalmanT[line, 2], skalmanT[line, 3], skalmanT[line, 4], skalmanT[line, 5], skalmanT[line, 6], skalmanT[line, 7], skalmanT[line, 8], skalmanT[line, 9],
          sma2T[line, 0], sma2T[line, 1], sma2T[line, 2], sma2T[line, 3], sma2T[line, 4], sma2T[line, 5], sma2T[line, 6], sma2T[line, 7], sma2T[line, 8], sma2T[line, 9],
          sma3T[line, 0], sma3T[line, 1], sma3T[line, 2], sma3T[line, 3], sma3T[line, 4], sma3T[line, 5], sma3T[line, 6], sma3T[line, 7], sma3T[line, 8], sma3T[line, 9],
          sma5T[line, 0], sma5T[line, 1], sma5T[line, 2], sma5T[line, 3], sma5T[line, 4], sma5T[line, 5], sma5T[line, 6], sma5T[line, 7], sma5T[line, 8], sma5T[line, 9],
          sma8T[line, 0], sma8T[line, 1], sma8T[line, 2], sma8T[line, 3], sma8T[line, 4], sma8T[line, 5], sma8T[line, 6], sma8T[line, 7], sma8T[line, 8], sma8T[line, 9],
          sma13T[line, 0], sma13T[line, 1], sma13T[line, 2], sma13T[line, 3], sma13T[line, 4], sma13T[line, 5], sma13T[line, 6], sma13T[line, 7], sma13T[line, 8], sma13T[line, 9],
          sma21T[line, 0], sma21T[line, 1], sma21T[line, 2], sma21T[line, 3], sma21T[line, 4], sma21T[line, 5], sma21T[line, 6], sma21T[line, 7], sma21T[line, 8], sma21T[line, 9],
          sma34T[line, 0], sma34T[line, 1], sma34T[line, 2], sma34T[line, 3], sma34T[line, 4], sma34T[line, 5], sma34T[line, 6], sma34T[line, 7], sma34T[line, 8], sma34T[line, 9],
          sma55T[line, 0], sma55T[line, 1], sma55T[line, 2], sma55T[line, 3], sma55T[line, 4], sma55T[line, 5], sma55T[line, 6], sma55T[line, 7], sma55T[line, 8], sma55T[line, 9],
          sma89T[line, 0], sma89T[line, 1], sma89T[line, 2], sma89T[line, 3], sma89T[line, 4], sma89T[line, 5], sma89T[line, 6], sma89T[line, 7], sma89T[line, 8], sma89T[line, 9],
          sma144T[line, 0], sma144T[line, 1], sma144T[line, 2], sma144T[line, 3], sma144T[line, 4], sma144T[line, 5], sma144T[line, 6], sma144T[line, 7], sma144T[line, 8], sma144T[line, 9],
          sep=';', file=f)
f.close()

delta = np.mean(np.abs(sT[:-2, :] - sT[1:-1, :]))
print(delta)

agmntCount = 5000
mu, sigma = 0, delta / 10
np.random.seed(0)

for agmnt_index in range(1, agmntCount + 1):
    agmntdT = np.zeros((np.size(sT[:, 0]), 120))
    agmntdT[:, 0:10] = sT + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 10:20] = skalmanT + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 20:30] = sma2T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 30:40] = sma3T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 40:50] = sma5T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 50:60] = sma8T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 60:70] = sma13T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 70:80] = sma21T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 80:90] = sma34T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 90:100] = sma55T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 100:110] = sma89T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))
    agmntdT[:, 110:120] = sma144T + np.random.normal(mu, sigma, (np.size(sT[:, 0]), np.size(sT[0, :])))

    f = open(path.join(outpath, "data/data_T_{0}.csv".format(agmnt_index)), 'w+')
    print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;ma2T_0;ma2T_1;ma2T_2;ma2T_3;ma2T_4;ma2T_5;ma2T_6;ma2T_7;ma2T_8;ma2T_9;ma3T_0;ma3T_1;ma3T_2;ma3T_3;ma3T_4;ma3T_5;ma3T_6;ma3T_7;ma3T_8;ma3T_9;ma5T_0;ma5T_1;ma5T_2;ma5T_3;ma5T_4;ma5T_5;ma5T_6;ma5T_7;ma5T_8;ma5T_9;ma8T_0;ma8T_1;ma8T_2;ma8T_3;ma8T_4;ma8T_5;ma8T_6;ma8T_7;ma8T_8;ma8T_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;ma21T_0;ma21T_1;ma21T_2;ma21T_3;ma21T_4;ma21T_5;ma21T_6;ma21T_7;ma21T_8;ma21T_9;ma34T_0;ma34T_1;ma34T_2;ma34T_3;ma34T_4;ma34T_5;ma34T_6;ma34T_7;ma34T_8;ma34T_9;ma55T_0;ma55T_1;ma55T_2;ma55T_3;ma55T_4;ma55T_5;ma55T_6;ma55T_7;ma55T_8;ma55T_9;ma89T_0;ma89T_1;ma89T_2;ma89T_3;ma89T_4;ma89T_5;ma89T_6;ma89T_7;ma89T_8;ma89T_9;ma144T_0;ma144T_1;ma144T_2;ma144T_3;ma144T_4;ma144T_5;ma144T_6;ma144T_7;ma144T_8;ma144T_9;", file=f)
    for line in range(0, len(T[:, 0])):
        print(N[line], sMode[line],
              agmntdT[line, 0], agmntdT[line, 1], agmntdT[line, 2], agmntdT[line, 3], agmntdT[line, 4], agmntdT[line, 5], agmntdT[line, 6], agmntdT[line, 7], agmntdT[line, 8], agmntdT[line, 9],
              agmntdT[line, 10], agmntdT[line, 11], agmntdT[line, 12], agmntdT[line, 13], agmntdT[line, 14], agmntdT[line, 15], agmntdT[line, 16], agmntdT[line, 17], agmntdT[line, 18], agmntdT[line, 19],
              agmntdT[line, 20], agmntdT[line, 21], agmntdT[line, 22], agmntdT[line, 23], agmntdT[line, 24], agmntdT[line, 25], agmntdT[line, 26], agmntdT[line, 27], agmntdT[line, 28], agmntdT[line, 29],
              agmntdT[line, 30], agmntdT[line, 31], agmntdT[line, 32], agmntdT[line, 33], agmntdT[line, 34], agmntdT[line, 35], agmntdT[line, 36], agmntdT[line, 37], agmntdT[line, 38], agmntdT[line, 39],
              agmntdT[line, 40], agmntdT[line, 41], agmntdT[line, 42], agmntdT[line, 43], agmntdT[line, 44], agmntdT[line, 45], agmntdT[line, 46], agmntdT[line, 47], agmntdT[line, 48], agmntdT[line, 49],
              agmntdT[line, 50], agmntdT[line, 51], agmntdT[line, 52], agmntdT[line, 53], agmntdT[line, 54], agmntdT[line, 55], agmntdT[line, 56], agmntdT[line, 57], agmntdT[line, 58], agmntdT[line, 59],
              agmntdT[line, 60], agmntdT[line, 61], agmntdT[line, 62], agmntdT[line, 63], agmntdT[line, 64], agmntdT[line, 65], agmntdT[line, 66], agmntdT[line, 67], agmntdT[line, 68], agmntdT[line, 69],
              agmntdT[line, 70], agmntdT[line, 71], agmntdT[line, 72], agmntdT[line, 73], agmntdT[line, 74], agmntdT[line, 75], agmntdT[line, 76], agmntdT[line, 77], agmntdT[line, 78], agmntdT[line, 79],
              agmntdT[line, 80], agmntdT[line, 81], agmntdT[line, 82], agmntdT[line, 83], agmntdT[line, 84], agmntdT[line, 85], agmntdT[line, 86], agmntdT[line, 87], agmntdT[line, 88], agmntdT[line, 89],
              agmntdT[line, 90], agmntdT[line, 91], agmntdT[line, 92], agmntdT[line, 93], agmntdT[line, 94], agmntdT[line, 95], agmntdT[line, 96], agmntdT[line, 97], agmntdT[line, 98], agmntdT[line, 99],
              agmntdT[line, 100], agmntdT[line, 101], agmntdT[line, 102], agmntdT[line, 103], agmntdT[line, 104], agmntdT[line, 105], agmntdT[line, 106], agmntdT[line, 107], agmntdT[line, 108], agmntdT[line, 109],
              agmntdT[line, 110], agmntdT[line, 111], agmntdT[line, 112], agmntdT[line, 113], agmntdT[line, 114], agmntdT[line, 115], agmntdT[line, 116], agmntdT[line, 117], agmntdT[line, 118], agmntdT[line, 119],
              sep=';', file=f)
    f.close()
