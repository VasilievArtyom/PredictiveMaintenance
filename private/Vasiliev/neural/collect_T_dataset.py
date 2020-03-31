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
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
                        ('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

#kalman filter
kalmanT = np.zeros((len(T[:,0]), 10))
for blc_id in range(0, 10):
	kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
						transition_covariance=0.01 * np.eye(2))
	states_pred = kf.em(T[:,blc_id]).smooth(T[:,blc_id])[0]
	kalmanT[:, blc_id] = states_pred[:, 0]

#moving average
ma_lags = [2, 3, 5, 8, 13]
maT=np.zeros((len(ma_lags), len(T[:,0]), 10))
for i, ma_lag in enumerate(ma_lags):
	maT[i,0:ma_lag,:].fill(np.nan)
	for j in range(ma_lag, len(T[:,0])):
		maT[i,j,:] = np.mean(T[j-ma_lag:j,:], axis=0)

f = open('data/RAWdataset.csv','w+')
print("N;Mode;T_0;T_1;T_2;T_3;T_4;T_5;T_6;T_7;T_8;T_9;kfT_0;kfT_1;kfT_2;kfT_3;kfT_4;kfT_5;kfT_6;kfT_7;kfT_8;kfT_9;ma2T_0;ma2T_1;ma2T_2;ma2T_3;ma2T_4;ma2T_5;ma2T_6;ma2T_7;ma2T_8;ma2T_9;ma3T_0;ma3T_1;ma3T_2;ma3T_3;ma3T_4;ma3T_5;ma3T_6;ma3T_7;ma3T_8;ma3T_9;ma5T_0;ma5T_1;ma5T_2;ma5T_3;ma5T_4;ma5T_5;ma5T_6;ma5T_7;ma5T_8;ma5T_9;ma8T_0;ma8T_1;ma8T_2;ma8T_3;ma8T_4;ma8T_5;ma8T_6;ma8T_7;ma8T_8;ma8T_9;ma13T_0;ma13T_1;ma13T_2;ma13T_3;ma13T_4;ma13T_5;ma13T_6;ma13T_7;ma13T_8;ma13T_9;"
	, file=f)
for line in range(0, len(T[:,0])):
	print(N[line, 0], Mode[line],
		T[line, 0], T[line, 1], T[line, 2], T[line, 3], T[line, 4], T[line, 5], T[line, 6], T[line, 7], T[line, 8], T[line, 9], 
		kalmanT[line, 0], kalmanT[line, 1], kalmanT[line, 2], kalmanT[line, 3], kalmanT[line, 4], kalmanT[line, 5], kalmanT[line, 6], kalmanT[line, 7], kalmanT[line, 8], kalmanT[line, 9],
		maT[0,line, 0], maT[0,line, 1], maT[0,line, 2], maT[0,line, 3], maT[0,line, 4], maT[0,line, 5], maT[0,line, 6], maT[0,line, 7], maT[0,line, 8], maT[0,line, 9],
		maT[1,line, 0], maT[1,line, 1], maT[1,line, 2], maT[1,line, 3], maT[1,line, 4], maT[1,line, 5], maT[1,line, 6], maT[1,line, 7], maT[1,line, 8], maT[1,line, 9],
		maT[2,line, 0], maT[2,line, 1], maT[2,line, 2], maT[2,line, 3], maT[2,line, 4], maT[2,line, 5], maT[2,line, 6], maT[2,line, 7], maT[2,line, 8], maT[2,line, 9],
		maT[3,line, 0], maT[3,line, 1], maT[3,line, 2], maT[3,line, 3], maT[3,line, 4], maT[3,line, 5], maT[3,line, 6], maT[3,line, 7], maT[3,line, 8], maT[3,line, 9],
		maT[4,line, 0], maT[4,line, 1], maT[4,line, 2], maT[4,line, 3], maT[4,line, 4], maT[4,line, 5], maT[4,line, 6], maT[4,line, 7], maT[4,line, 8], maT[4,line, 9],
		sep=';', file=f)
