import math
import numpy as np
from numpy import *
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
import scipy.stats as stats
from scipy import signal

plt.rc('text', usetex=True)

inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
T_n_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$']

upper_threshold = []
lower_threshold = []

for n in range(0, 10):
	upper_threshold_perblock = []
	lower_threshold_perblock = []
	for t in range(1, len(T[:, n])):
		if (S[t, n] and not S[t-1, n]):
			lower_threshold.append(T[t, n])
			lower_threshold_perblock.append(T[t, n])
		if (not S[t, n] and S[t-1, n]):
			upper_threshold.append(T[t, n])
			upper_threshold_perblock.append(T[t, n])
	upper_threshold_perblock = np.array(upper_threshold_perblock)
	lower_threshold_perblock = np.array(lower_threshold_perblock)
	upper_threshold_perblock_mean = np.mean(upper_threshold_perblock)
	lower_threshold_perblock_mean = np.mean(lower_threshold_perblock)

	upper_threshold_perblock_err = np.std(upper_threshold_perblock, ddof=1) / np.sqrt(len(upper_threshold_perblock))
	lower_threshold_perblock_err = np.std(lower_threshold_perblock, ddof=1) / np.sqrt(len(lower_threshold_perblock))
	print("#threshold \pm error per block"+str(n+1))
	print(np.round(upper_threshold_perblock_mean, decimals=2), np.round(upper_threshold_perblock_err, decimals=2))
	print(np.round(lower_threshold_perblock_mean, decimals=2), np.round(lower_threshold_perblock_err, decimals=2))

upper_threshold = np.array(upper_threshold)
lower_threshold = np.array(lower_threshold)

upper_threshold_mean = np.mean(upper_threshold)
lower_threshold_mean = np.mean(lower_threshold)

upper_threshold_err = np.std(upper_threshold, ddof=1) / np.sqrt(len(upper_threshold))
lower_threshold_err = np.std(lower_threshold, ddof=1) / np.sqrt(len(lower_threshold))

print("#threshold \pm error ")
print(np.round(upper_threshold_mean, decimals=2), np.round(upper_threshold_err, decimals=2))
print(np.round(lower_threshold_mean, decimals=2), np.round(lower_threshold_err, decimals=2))