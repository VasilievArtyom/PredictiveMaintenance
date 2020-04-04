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


def read_dtaset_by_index(index):
    inpath = "../neural/data/"
    currentfile = path.join(inpath, "data_T_{0}.csv".format(index))
    # Read from file
    strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_),
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
    # N, _Mode, _T, _kalmanT, _ma2T, _ma3T, _ma5T, _ma8T, _ma13T
    return np.loadtxt(currentfile, unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


# Read unaugmented dataset
N, Mode, T, kalmanT, ma2T, ma3T, ma5T, ma8T, ma13T, ma21T, ma34T, ma55T, ma89T, ma144T = read_dtaset_by_index(0)
print(kalmanT[:, 0])

# fit model
p = range(0, 8)
q = range(0, 8)
d = range(0, 5)
pdq = list(itertools.product(p, d, q))
warnings.filterwarnings("ignore")
for param in pdq:
    for sparam in pdq:
        try:
            print("###############################")
            print(param, sparam)
            print("###############################")
            sprm = np.array(sparam)
            model = sm.tsa.statespace.SARIMAX(kalmanT[:, 0],
                                              order=param,
                                              seasonal_order=(sprm[0], sprm[1], sprm[2], 50),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
            print("ffffffffffffffffffffffffffffffffffffffff")
            results = model.fit()
            f = open('hyperparam_Kalman.txt', 'a+')
            print('ARIMA{}{} - AIC:{}'.format(param, sparam, results.aic), file=f)
            f.close()
        except:
            continue