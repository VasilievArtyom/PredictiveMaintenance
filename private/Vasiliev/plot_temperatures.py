import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf


plt.rc('text', usetex=True)

outpath = "../../plots/temperatures"
inpath = "../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
						('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile),
	unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)
T_n_labels = [r'$T_1$', r'$T_2$', r'$T_3$', r'$T_4$', r'$T_5$', r'$T_6$', r'$T_7$', r'$T_8$', r'$T_9$', r'$T_{10}$']

# # Draw Raw temperature plots
# for i in range(0, 10):
# 	fig, ax = plt.subplots(figsize=(10, 5))
# 	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# 	# colors = ['b', 'w', 'w', 'w', 'w', 'w', 'w']
# 	maxval = np.amax(T[:, i])
# 	minval = np.amin(T[:, i])
# 	dif = maxval - minval
# 	for pnt in range(0, np.size(N[:,0])):
# 		ax.scatter(N[pnt,0], T[pnt, i], marker='.', color=colors[Mode[pnt]])
# 	crushflag = np.invert(S[:, i])
# 	crushflag = crushflag.astype(int)
# 	ax.fill_between(N[:,0], minval, minval + dif * crushflag,
#                 facecolor='maroon',
#                 interpolate=True,
#                 zorder=0,
#                 alpha=0.2)
# 	plt.grid()
# 	plt.ylabel(T_n_labels[i])
# 	plt.xlabel(r'$n$')
# 	ax.xaxis.grid(b=True, which='both')
# 	ax.yaxis.grid(b=True, which='both')
# 	plt.tight_layout()
# 	plt.draw()
# 	fig.savefig(path.join(outpath, "T_{0}_raw.png".format(i+1)))
# 	plt.clf()

# # Draw Raw temperature plots
# for i in range(0, 10):
# 	fig, ax = plt.subplots(figsize=(10, 5))
# 	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# 	# colors = ['b', 'w', 'w', 'w', 'w', 'w', 'w']
# 	maxval = np.amax(T[:, i])
# 	minval = np.amin(T[:, i])
# 	dif = maxval - minval
# 	frm = np.size(N[:,0]) - 500
# 	for pnt in range(frm, np.size(N[:,0])):
# 		ax.scatter(N[pnt,0], T[pnt, i], marker='.', color=colors[Mode[pnt]])
# 	crushflag = np.invert(S[frm:, i])
# 	crushflag = crushflag.astype(int)
# 	ax.fill_between(N[frm:,0], minval, minval + dif * crushflag,
#                 facecolor='maroon',
#                 interpolate=True,
#                 zorder=0,
#                 alpha=0.5)
# 	plt.grid()
# 	plt.ylabel(T_n_labels[i])
# 	plt.xlabel(r'$n$')
# 	ax.xaxis.grid(b=True, which='both')
# 	ax.yaxis.grid(b=True, which='both')
# 	plt.tight_layout()
# 	plt.draw()
# 	fig.savefig(path.join(outpath, "T_{0}_raw_tail.png".format(i+1)))
# 	plt.clf()



# fig, ax = plt.subplots(figsize=(10, 5))
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# meanT = np.mean(T, axis=1)
# maxval = np.amax(meanT)
# minval = np.amin(meanT)
# dif = maxval - minval
# for pnt in range(0, np.size(N[:,0])):
# 	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
# crushflag = np.invert(System_State[:])
# crushflag = crushflag.astype(int)
# ax.fill_between(N[:,0], minval, minval + dif * crushflag,
#                facecolor='maroon',
#                interpolate=True,
#                zorder=0,
#                alpha=0.1)
# plt.grid()
# plt.ylabel(r'$\overline{T}$')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# plt.tight_layout()
# plt.draw()
# fig.savefig(path.join(outpath, "T_mean.png"))
# plt.clf()


# fig, ax = plt.subplots(figsize=(10, 5))
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# meanT = np.amax(T, axis=1)
# maxval = np.amax(meanT)
# minval = np.amin(meanT)
# dif = maxval - minval
# for pnt in range(0, np.size(N[:,0])):
# 	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
# crushflag = np.invert(System_State[:])
# crushflag = crushflag.astype(int)
# ax.fill_between(N[:,0], minval, minval + dif * crushflag,
#                facecolor='maroon',
#                interpolate=True,
#                zorder=0,
#                alpha=0.1)
# plt.grid()
# plt.ylabel(r'$T^{(max)}$')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# plt.tight_layout()
# plt.draw()
# fig.savefig(path.join(outpath, "T_max_per_block.png"))
# plt.clf()


# fig, ax = plt.subplots(figsize=(10, 5))
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# meanT = np.amin(T, axis=1)
# maxval = np.amax(meanT)
# minval = np.amin(meanT)
# dif = maxval - minval
# for pnt in range(0, np.size(N[:,0])):
# 	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
# crushflag = np.invert(System_State[:])
# crushflag = crushflag.astype(int)
# ax.fill_between(N[:,0], minval, minval + dif * crushflag,
#                facecolor='maroon',
#                interpolate=True,
#                zorder=0,
#                alpha=0.1)
# plt.grid()
# plt.ylabel(r'$T^{(min)}$')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# plt.tight_layout()
# plt.draw()
# fig.savefig(path.join(outpath, "T_min_per_block.png"))
# plt.clf()

# fig, ax = plt.subplots(figsize=(10, 5))
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# meanT = np.amax(T, axis=1) - np.amin(T, axis=1)
# maxval = np.amax(meanT)
# minval = np.amin(meanT)
# dif = maxval - minval
# for pnt in range(0, np.size(N[:,0])):
# 	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
# crushflag = np.invert(System_State[:])
# crushflag = crushflag.astype(int)
# ax.fill_between(N[:,0], minval, minval + dif * crushflag,
#                facecolor='maroon',
#                interpolate=True,
#                zorder=0,
#                alpha=0.1)
# plt.grid()
# plt.ylabel(r'$T^{(max)} - T^{(min)}$')
# plt.xlabel(r'$n$')
# ax.xaxis.grid(b=True, which='both')
# ax.yaxis.grid(b=True, which='both')
# plt.tight_layout()
# plt.draw()
# fig.savefig(path.join(outpath, "T_diff.png"))
# plt.clf()


fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.mean(T, axis=1)
maxval = np.amax(meanT)
minval = np.amin(meanT)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])):
	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$\overline{T}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_mean.png"))
plt.clf()


fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.amax(T, axis=1)
maxval = np.amax(meanT)
minval = np.amin(meanT)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])):
	ax.scatter(N[pnt,0], meanT[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$T^{(max)}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_max_per_block.png"))
plt.clf()


fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.mean(T, axis=1)
dot_mean_T = meanT[1:] - meanT[0:-1]
maxval = np.amax(dot_mean_T)
minval = np.amin(dot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-1):
	ax.scatter(N[pnt,0], dot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$\dot{T}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_dot.png"))
plt.clf()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.mean(T, axis=1)
dot_mean_T = meanT[1:] - meanT[0:-1]
dot_mean_T = np.abs(dot_mean_T)
maxval = np.amax(dot_mean_T)
minval = np.amin(dot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-1):
	ax.scatter(N[pnt,0], dot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$|\dot{T}|$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_dot_abs.png"))
plt.clf()


fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.mean(T, axis=1)
ddot_mean_T = meanT[0:-2] + meanT[2] - 2 * meanT[1:-1] 
maxval = np.amax(ddot_mean_T)
minval = np.amin(ddot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-2):
	ax.scatter(N[pnt,0], ddot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$\ddot{T}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_dot_dot.png"))
plt.clf()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.mean(T, axis=1)
ddot_mean_T = meanT[0:-2] + meanT[2] - 2 * meanT[1:-1]
ddot_mean_T = np.abs(ddot_mean_T)
maxval = np.amax(ddot_mean_T)
minval = np.amin(ddot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-2):
	ax.scatter(N[pnt,0], ddot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$|\ddot{T}|$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_dot_dot_abs.png"))
plt.clf()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.amax(T, axis=1)
dot_mean_T = meanT[1:] - meanT[0:-1]
maxval = np.amax(dot_mean_T)
minval = np.amin(dot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-1):
	ax.scatter(N[pnt,0], dot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$\dot{T^{(max)}}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T__max_dot.png"))
plt.clf()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.amax(T, axis=1)
dot_mean_T = meanT[1:] - meanT[0:-1]
dot_mean_T = np.abs(dot_mean_T)
maxval = np.amax(dot_mean_T)
minval = np.amin(dot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-1):
	ax.scatter(N[pnt,0], dot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$|\dot{T^{(max)}}|$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_max_dot_abs.png"))
plt.clf()


fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.amax(T, axis=1)
ddot_mean_T = meanT[0:-2] + meanT[2] - 2 * meanT[1:-1] 
maxval = np.amax(ddot_mean_T)
minval = np.amin(ddot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-2):
	ax.scatter(N[pnt,0], ddot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$\ddot{T^{(max)}}$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_max_dot_dot.png"))
plt.clf()

fig, ax = plt.subplots(figsize=(5, 5))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
meanT = np.amax(T, axis=1)
ddot_mean_T = meanT[0:-2] + meanT[2] - 2 * meanT[1:-1]
ddot_mean_T = np.abs(ddot_mean_T)
maxval = np.amax(ddot_mean_T)
minval = np.amin(ddot_mean_T)
dif = maxval - minval
for pnt in range(2000, np.size(N[:,0])-2):
	ax.scatter(N[pnt,0], ddot_mean_T[pnt], marker='.', color=colors[Mode[pnt]])
crushflag = np.invert(System_State[2000:])
crushflag = crushflag.astype(int)
ax.fill_between(N[2000:,0], minval, minval + dif * crushflag,
               facecolor='maroon',
               interpolate=True,
               zorder=0,
               alpha=0.1)
plt.grid()
plt.ylabel(r'$|\ddot{T^{(max)}}|$')
plt.xlabel(r'$n$')
ax.xaxis.grid(b=True, which='both')
ax.yaxis.grid(b=True, which='both')
plt.tight_layout()
plt.draw()
fig.savefig(path.join(outpath, "T_max_dot_dot_abs.png"))
plt.clf()