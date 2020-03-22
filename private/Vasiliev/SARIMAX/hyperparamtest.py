import math
import numpy as np
from numpy import *
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import itertools
import warnings


plt.rc('text', usetex=True)

outpath = "../../../plots/temperatures"
inpath = "../../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
                        ('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)



# fit model
p = q = range(0, 47)
d = [0]
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore")
f = open('hyperparam.txt','w') 
for param in pdq:
	try:
		print("###############################")
		print(param)
		print("###############################")
		model = sm.tsa.statespace.SARIMAX(T[:,0],
										  order=param,
										  seasonal_order=(0,0,0, 48),
										  enforce_stationarity=False,
										  enforce_invertibility=False)
		results = model.fit()
		f = open('hyperparam.txt','a+') 
		print('ARIMA{} - AIC:{}'.format(param, results.aic), file=f)
		f.close()
	except:
		continue




# final_model = sm.tsa.statespace.SARIMAX(fullW[65:],
# 										order=(2, 0, 14),
# 										seasonal_order=(0,0,0, 12),
# 										enforce_stationarity=False,
# 										enforce_invertibility=False)
# results = final_model.fit()
# print(results.summary().tables[1])

# pred = results.get_prediction(end=1043, dynamic=False)
# pred_vals = pred.predicted_mean
# pred_ci = pred.conf_int()

# extended_n = np.arange(0, 1044, 1)

# # Draw naive prediction plot
# fig, ax = plt.subplots(figsize=(8, 3.8))
# ax.scatter(extended_n[1024:], pred_vals[1024:], 
# 			marker='+',
# 			color='crimson',
# 			label='Prediction',
# 			zorder=10)
# ax.fill_between(extended_n[1024:],
#                 pred_ci[1024, 0],
#                 pred_ci[1024:, 1],
#                 facecolor='gainsboro', 
#                 label='Confidence interval',
#                 interpolate=True,
#                 zorder=0)
# ax.plot(fulln[800:], fullW[800:],  ls='-', label='Raw signal', zorder=5)
# plt.grid()
# plt.ylabel(r'$W_{n}$')
# plt.xlabel(r'$t_n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# ax.legend(loc='upper left', frameon=True)
# plt.draw()
# fig.savefig(path.join(outpath, "prediction.png"))
# plt.clf()

# f = open('prediction.txt','w')
# print('#timestamp, value, Confidence interval bounds', file=f)
# for index in range(1024, 1043):
# 	print(extended_n[index], pred_vals[index], pred_ci[index, 0], pred_ci[index, 1], file=f)
