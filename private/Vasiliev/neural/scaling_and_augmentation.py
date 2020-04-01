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
strdatatype = np.dtype([('N', np.int_), ('Mode', np.int_ ),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('ma2T', np.float_, (10,)),
                        ('ma3T', np.float_, (10,)),
                        ('ma5T', np.float_, (10,)),
                        ('ma8T', np.float_, (10,)),
                        ('ma13T', np.float_, (10,))])
N, Mode, T, kalmanT, ma2T, ma3T, ma5T, ma8T, ma13T = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
#map from
print(np.amax(T), np.amin(T), np.mean(T))
print(np.amax(kalmanT), np.amin(kalmanT), np.mean(kalmanT))
print(np.amax(ma2T), np.amin(ma2T), np.mean(ma2T))
print(np.amax(ma3T), np.amin(ma3T), np.mean(ma3T))
print(np.amax(ma5T), np.amin(ma5T), np.mean(ma5T))
print(np.amax(ma8T), np.amin(ma8T), np.mean(ma8T))
print(np.amax(ma13T), np.amin(ma13T), np.mean(ma13T))
print(np.amax(Mode), np.amin(Mode), np.mean(Mode))

#map to [-1, 1] with zero mean
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
sma2T[:2:].fill(np.nan)
sma3T[:3:].fill(np.nan)
sma5T[:5:].fill(np.nan)
sma8T[:8:].fill(np.nan)
sma13T[:13:].fill(np.nan)


fill_latency_for_T = 0.90
for blc_id in range(0, 10):
	sT[:, blc_id]  = scale_sinle_series(T[:, blc_id] , np.amax(T[:, blc_id]), np.amin(T[:, blc_id]), np.mean(T[:, blc_id]), fill_latency_for_T)
	skalmanT[:, blc_id] = scale_sinle_series(kalmanT[:, blc_id], np.amax(kalmanT[:, blc_id]), np.amin(kalmanT[:, blc_id]), np.mean(kalmanT[:, blc_id]), fill_latency_for_T)
	sma2T[2:, blc_id] = scale_sinle_series(ma2T[2:, blc_id], np.amax(ma2T[2:, blc_id]), np.amin(ma2T[2:, blc_id]), np.mean(ma2T[2:, blc_id]), fill_latency_for_T)
	sma3T[3:, blc_id] = scale_sinle_series(ma3T[3:, blc_id], np.amax(ma3T[3:, blc_id]), np.amin(ma3T[3:, blc_id]), np.mean(ma3T[3:, blc_id]), fill_latency_for_T)
	sma5T[5:, blc_id] = scale_sinle_series(ma5T[5:, blc_id], np.amax(ma5T[5:, blc_id]), np.amin(ma5T[5:, blc_id]), np.mean(ma5T[5:, blc_id]), fill_latency_for_T)
	sma8T[8:, blc_id] = scale_sinle_series(ma8T[8:, blc_id], np.amax(ma8T[8:, blc_id]), np.amin(ma8T[8:, blc_id]), np.mean(ma8T[8:, blc_id]), fill_latency_for_T)
	sma13T[13:, blc_id] = scale_sinle_series(ma13T[13:, blc_id], np.amax(ma13T[13:, blc_id]), np.amin(ma13T[13:, blc_id]), np.mean(ma13T[13:, blc_id]), fill_latency_for_T)

sMode = scale_sinle_series(Mode, np.amax(Mode), np.amin(Mode), np.mean(Mode), 1.0)
print("<--------------------------------->")
print(np.amax(sT), np.amin(sT), np.mean(sT))
print(np.amax(skalmanT), np.amin(skalmanT), np.mean(skalmanT))
print(np.amax(sma2T), np.amin(sma2T), np.mean(sma2T))
print(np.amax(sma3T), np.amin(sma3T), np.mean(sma3T))
print(np.amax(sma5T), np.amin(sma5T), np.mean(sma5T))
print(np.amax(sma8T), np.amin(sma8T), np.mean(sma8T))
print(np.amax(sma13T), np.amin(sma13T), np.mean(sma13T))
print(np.amax(sMode), np.amin(sMode), np.mean(sMode))

f = open('data/data_T_0.csv','w+')
print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;ma2T_0;ma2T_1;ma2T_2;ma2T_3;ma2T_4;ma2T_5;ma2T_6;ma2T_7;ma2T_8;ma2T_9;ma3T_0;ma3T_1;ma3T_2;ma3T_3;ma3T_4;ma3T_5;ma3T_6;ma3T_7;ma3T_8;ma3T_9;ma5T_0;ma5T_1;ma5T_2;ma5T_3;ma5T_4;ma5T_5;ma5T_6;ma5T_7;ma5T_8;ma5T_9;ma8T_0;ma8T_1;ma8T_2;ma8T_3;ma8T_4;ma8T_5;ma8T_6;ma8T_7;ma8T_8;ma8T_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;"
	, file=f)
for line in range(0, len(T[:,0])):
	print(N[line], sMode[line],
		sT[line, 0], sT[line, 1], sT[line, 2], sT[line, 3], sT[line, 4], sT[line, 5], sT[line, 6], sT[line, 7], sT[line, 8], sT[line, 9], 
		skalmanT[line, 0], skalmanT[line, 1], skalmanT[line, 2], skalmanT[line, 3], skalmanT[line, 4], skalmanT[line, 5], skalmanT[line, 6], skalmanT[line, 7], skalmanT[line, 8], skalmanT[line, 9],
		sma2T[line, 0], sma2T[line, 1], sma2T[line, 2], sma2T[line, 3], sma2T[line, 4], sma2T[line, 5], sma2T[line, 6], sma2T[line, 7], sma2T[line, 8], sma2T[line, 9],
		sma3T[line, 0], sma3T[line, 1], sma3T[line, 2], sma3T[line, 3], sma3T[line, 4], sma3T[line, 5], sma3T[line, 6], sma3T[line, 7], sma3T[line, 8], sma3T[line, 9],
		sma5T[line, 0], sma5T[line, 1], sma5T[line, 2], sma5T[line, 3], sma5T[line, 4], sma5T[line, 5], sma5T[line, 6], sma5T[line, 7], sma5T[line, 8], sma5T[line, 9],
		sma8T[line, 0], sma8T[line, 1], sma8T[line, 2], sma8T[line, 3], sma8T[line, 4], sma8T[line, 5], sma8T[line, 6], sma8T[line, 7], sma8T[line, 8], sma8T[line, 9],
		sma13T[line, 0], sma13T[line, 1], sma13T[line, 2], sma13T[line, 3], sma13T[line, 4], sma13T[line, 5], sma13T[line, 6], sma13T[line, 7], sma13T[line, 8], sma13T[line, 9],
		sep=';', file=f)
f.close()

delta = np.mean(np.abs(sT[:-2, :] - sT[1:-1, :]))
print(delta)

agmntCount=5000
mu, sigma = 0, delta / 10
np.random.seed(0)

for agmnt_index in range(1, agmntCount+1):
	agmntdT=np.zeros((np.size(sT[:,0]), 70))
	agmntdT[:,0:10] = sT + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,10:20] = skalmanT + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,20:30] = sma2T + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,30:40] = sma3T + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,40:50] = sma5T + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,50:60] = sma8T + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	agmntdT[:,60:70] = sma13T + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))
	f = open(path.join(outpath, "data/data_T_{0}.csv".format(agmnt_index)),'w+')
	print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;ma2T_0;ma2T_1;ma2T_2;ma2T_3;ma2T_4;ma2T_5;ma2T_6;ma2T_7;ma2T_8;ma2T_9;ma3T_0;ma3T_1;ma3T_2;ma3T_3;ma3T_4;ma3T_5;ma3T_6;ma3T_7;ma3T_8;ma3T_9;ma5T_0;ma5T_1;ma5T_2;ma5T_3;ma5T_4;ma5T_5;ma5T_6;ma5T_7;ma5T_8;ma5T_9;ma8T_0;ma8T_1;ma8T_2;ma8T_3;ma8T_4;ma8T_5;ma8T_6;ma8T_7;ma8T_8;ma8T_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;"
	, file=f)
	for line in range(0, len(T[:,0])):
		print(N[line], sMode[line],
			agmntdT[line,0], agmntdT[line,1], agmntdT[line,2], agmntdT[line,3], agmntdT[line,4], agmntdT[line,5], agmntdT[line,6], agmntdT[line, 7], agmntdT[line,8], agmntdT[line,9],
			agmntdT[line,10], agmntdT[line,11], agmntdT[line,12], agmntdT[line,13], agmntdT[line,14], agmntdT[line,15], agmntdT[line,16], agmntdT[line,17], agmntdT[line,18], agmntdT[line,19],
			agmntdT[line,20], agmntdT[line,21], agmntdT[line,22], agmntdT[line,23], agmntdT[line,24], agmntdT[line,25], agmntdT[line,26], agmntdT[line,27], agmntdT[line,28], agmntdT[line,29],
			agmntdT[line,30], agmntdT[line,31], agmntdT[line,32], agmntdT[line,33], agmntdT[line,34], agmntdT[line,35], agmntdT[line,36], agmntdT[line,37], agmntdT[line,38], agmntdT[line,39],
			agmntdT[line,40], agmntdT[line,41], agmntdT[line,42], agmntdT[line,43], agmntdT[line,44], agmntdT[line,45], agmntdT[line,46], agmntdT[line,47], agmntdT[line,48], agmntdT[line,49],
			agmntdT[line,50], agmntdT[line,51], agmntdT[line,52], agmntdT[line,53], agmntdT[line,54], agmntdT[line,55], agmntdT[line,56], agmntdT[line,57], agmntdT[line,58], agmntdT[line,59],
			agmntdT[line,60], agmntdT[line,61], agmntdT[line,62], agmntdT[line,63], agmntdT[line,64], agmntdT[line,65], agmntdT[line,66], agmntdT[line,67], agmntdT[line,68], agmntdT[line,69],
			sep=';', file=f)
	f.close()