import math
import numpy as np
from numpy import *
from os import path
import os
import sys
import matplotlib.pyplot as plt

import keras
from keras.utils import Sequence
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import losses
from keras.utils import plot_model
from tensorflow.python.client import device_lib


plt.rc('text', usetex=True)
np.random.seed(0)  # Set a random seed for reproducibility

blc_id = 0
gpu_id = str(1)
if len(sys.argv) > 1:
    blc_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])

# <--------------------->
# Tunable

rnn_sequence_length = 300
cutFromTail = 60
cutFromHead = 144
max_pred_step = 60
# <--------------------->


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# <--------------------->
# Tunable
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# <--------------------->
# print(device_lib.list_local_devices())


def read_dtaset_by_index(index):
    inpath = "../data/"
    currentfile = path.join(inpath, "data_T_{0}.csv".format(index))
    # Read from file
    strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_, (5,)),
                            ('kalmanT', np.float_, (10,)),
                            ('kalmanT_dot', np.float_, (10,)),
                            ('rwavT', np.float_, (10,)),
                            ('ma13T', np.float_, (10,)),
                            ('ma55T', np.float_, (10,)),
                            ('ma144T', np.float_, (10,)),
                            ('S', np.float_, (10,)),
                            ('lfc', np.float_, (10,))])
    # N, Mode, kalmanT, kalmanT_dot, rwavT, ma13T, ma55T, ma144T, S, lfc
    return np.loadtxt(currentfile, unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


# Read unaugmented dataset
N, Mode, kalmanT, kalmanT_dot, rwavT, ma13T, ma55T, ma144T, S, lfc = read_dtaset_by_index(0)
# Collect full dataset
n_features = 13
l_b, r_b = cutFromHead, cutFromTail
N = N[l_b:-r_b]
ds = np.empty((10, len(N), n_features))
for _blc_id in range(0, 10):
    (ds[_blc_id, :, 0], ds[_blc_id, :, 1], ds[_blc_id, :, 2],
     ds[_blc_id, :, 3], ds[_blc_id, :, 4], ds[_blc_id, :, 5],
     ds[_blc_id, :, 6], ds[_blc_id, :, 7], ds[_blc_id, :, 8:13]) = (kalmanT[l_b:-r_b, _blc_id], kalmanT_dot[l_b:-r_b, _blc_id],
                                                                    rwavT[l_b:-r_b, _blc_id], ma13T[l_b:-r_b, _blc_id], ma55T[l_b:-r_b, _blc_id],
                                                                    ma144T[l_b:-r_b, _blc_id], S[l_b:-r_b, _blc_id], lfc[l_b:-r_b, _blc_id], Mode[l_b:-r_b, :])
S = S[l_b:-r_b, :]


def get_data_by_timestamp(_n, _blc_id, _pred_step):
    min_poss_n, max_poss_n = np.amin(N) + rnn_sequence_length, np.amax(N) - max_pred_step
    assert ((_n <= max_poss_n) and (_n >= min_poss_n)), "Out of bounds"
    index = int(np.where(N == _n)[0])

    X1_shape = (1, rnn_sequence_length, 13)
    X1 = np.zeros(shape=X1_shape, dtype=np.float16)
    X2_shape = (1, 13)
    X2 = np.zeros(shape=X2_shape, dtype=np.float16)

    Y1_shape = (1, 1)
    Y1 = np.zeros(shape=Y1_shape, dtype=np.float16)
    Y2_shape = (1, 1)
    Y2 = np.zeros(shape=Y2_shape, dtype=np.float16)

    idx = index + 1 - rnn_sequence_length
    X1[0, :, :] = ds[_blc_id, idx:idx + rnn_sequence_length, :]
    X2[0, :] = ds[_blc_id, idx + rnn_sequence_length - 1, :]

    Y1[0, 0] = ds[_blc_id, idx + rnn_sequence_length - 1, 0]
    Y2[0, 0] = ds[_blc_id, idx + _pred_step + rnn_sequence_length - 1, 6]

    return [X1, X2], [Y1, Y2]


def get_ground_true_by_timestamp(_n, _blc_id, _pred_step):
    min_poss_n, max_poss_n = np.amin(N) + rnn_sequence_length, np.amax(N) - max_pred_step
    assert ((_n <= max_poss_n) and (_n >= min_poss_n)), "Out of bounds"
    index = int(np.where(N == _n)[0])
    return S[index + _pred_step, _blc_id]


# ################# model difinition #########################################
recurrent_input = Input(shape=(rnn_sequence_length, 13), name='recurrent_input')
rec_layer_1 = LSTM(150, return_sequences=True)(recurrent_input)
rec_layer_2 = LSTM(64)(rec_layer_1)
recurrent_output = Dense(1, activation='tanh', name='recurrent_output')(rec_layer_2)

sequential_input = Input(shape=(13,), name='sequential_input')
x = keras.layers.concatenate([rec_layer_2, sequential_input])

x = Dense(128, activation='tanh')(x)
x = Dense(64, activation='tanh')(x)
x = Dense(32, activation='tanh')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[recurrent_input, sequential_input], outputs=[recurrent_output, main_output])
plot_model(model, to_file='model.png', show_shapes=True)
model.compile(optimizer='Adam',
              loss={'main_output': 'binary_crossentropy', 'recurrent_output': 'logcosh'},
              loss_weights={'main_output': 1., 'recurrent_output': 0.50})
#############################################################################


def perfotm_prediction_over_timestamp(_n, _blc_id, _pred_step):
    path_checkpoint = '../models/' + str(_blc_id) + '_binary_on_' + str(_pred_step) + '.keras'

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)
    [tmpX1, tmpX2], [Y1, Y2] = get_data_by_timestamp(_n, _blc_id, _pred_step)
    [pred1, pred2] = model.predict({'recurrent_input': tmpX1, 'sequential_input': tmpX2})
    return float(pred2[0, 0])


def count_mismatches(_from, _to, _blc_id, _pred_step, _level):
    current_count = 0
    for tmstmp in range(_from, _to + 1):
        true_val = get_ground_true_by_timestamp(tmstmp, _blc_id, _pred_step)
        pred_val = perfotm_prediction_over_timestamp(tmstmp, _blc_id, _pred_step)
        tmpval = 0.0
        if (pred_val > _level):
            tmpval = 1.0
        if (int(tmpval) != true_val):
            current_count += 1
        print(true_val, pred_val, current_count)
    return current_count


levels = np.empty((21, 10))
min_tmpstmp = rnn_sequence_length + 144
max_tmstmp = len(N) - max_pred_step
print(min_tmpstmp, max_tmstmp)

for pred_step in range(21, 22):
    f = open((str(blc_id) + '_binary_on_' + str(pred_step) + ".csv"), 'w+')
    print("N,GT,predict", file=f)
    for tmstmp in range(min_tmpstmp, max_tmstmp):
        true_val = get_ground_true_by_timestamp(tmstmp, blc_id, pred_step)
        pred_val = perfotm_prediction_over_timestamp(tmstmp, blc_id, pred_step)
        print(tmstmp, true_val, pred_val, sep=',', file=f)
