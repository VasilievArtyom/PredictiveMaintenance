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

currentfile = "data_T_0.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_ ),
                        ('T', np.float_, (10,)),
                        ('kalmanT', np.float_, (10,)),
                        ('ma2T', np.float_, (10,)),
                        ('ma3T', np.float_, (10,)),
                        ('ma5T', np.float_, (10,)),
                        ('ma8T', np.float_, (10,)),
                        ('ma13T', np.float_, (10,))])
N, Mode, T, kalmanT, ma2T, ma3T, ma5T, ma8T, ma13T = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

print(np.amax(T), np.amin(T), np.mean(T))
print(np.amax(kalmanT), np.amin(kalmanT), np.mean(kalmanT))
print(np.amax(ma2T), np.amin(ma2T), np.mean(ma2T))
print(np.amax(ma3T), np.amin(ma3T), np.mean(ma3T))
print(np.amax(ma5T), np.amin(ma5T), np.mean(ma5T))
print(np.amax(ma8T), np.amin(ma8T), np.mean(ma8T))
print(np.amax(ma13T), np.amin(ma13T), np.mean(ma13T))
print(np.amax(Mode), np.amin(Mode), np.mean(Mode))

