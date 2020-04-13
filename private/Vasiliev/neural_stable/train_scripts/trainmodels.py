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

# <--------------------->
# Tunable

agmntCount = 10
blc_id = 0
pred_step = 1
gpu_id = str(1)

rnn_sequence_length = 300
cutFromTail = 60
cutFromHead = 144
max_pred_step = 60
# <--------------------->


if len(sys.argv) > 1:
    blc_id = int(sys.argv[1])
    pred_step = int(sys.argv[2])
    gpu_id = str(sys.argv[3])


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# <--------------------->
# Tunable
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
batch_size = 520
# <--------------------->
print(device_lib.list_local_devices())


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
# Alloc and read agmntCount augmented copies, collect full dataset
n_features = 13
ds = np.empty((agmntCount + 1, len(kalmanT[:, 0]), n_features))
(ds[0, :, 0], ds[0, :, 1], ds[0, :, 2], ds[0, :, 3], ds[0, :, 4],
    ds[0, :, 5], ds[0, :, 6], ds[0, :, 7], ds[0, :, 8:13]) = (kalmanT[:, blc_id], kalmanT_dot[:, blc_id],
                                                              rwavT[:, blc_id], ma13T[:, blc_id], ma55T[:, blc_id],
                                                              ma144T[:, blc_id], S[:, blc_id], lfc[:, blc_id], Mode[:, :])
for agmnt_index in range(1, agmntCount + 1):
    _N, _Mode, _kalmanT, _kalmanT_dot, _rwavT, _ma13T, _ma55T, _ma144T, _S, _lfc = read_dtaset_by_index(agmnt_index)
    (ds[agmnt_index, :, 0], ds[agmnt_index, :, 1], ds[agmnt_index, :, 2], ds[agmnt_index, :, 3], ds[agmnt_index, :, 4],
     ds[agmnt_index, :, 5], ds[agmnt_index, :, 6], ds[agmnt_index, :, 7], ds[agmnt_index, :, 8:13]) = (_kalmanT[:, blc_id], _kalmanT_dot[:, blc_id],
                                                                                                       _rwavT[:, blc_id], _ma13T[:, blc_id], _ma55T[:, blc_id],
                                                                                                       _ma144T[:, blc_id], _S[:, blc_id], _lfc[:, blc_id], _Mode[:, :])


# boundaries
l_b, r_b = cutFromHead, cutFromTail
len_data = len(ds[0, l_b:-r_b, 0])

len_test = int(rnn_sequence_length * 1.25)
len_train = len_data - len_test
print(l_b, r_b)
print("len_data = ", len_data)
print("len_test = ", len_test)
print("len_train = ", len_train)

ds_train = ds[:, l_b:l_b + len_train, :]
ds_test = ds[:, -(r_b + len_test):-r_b, :]

print("ds: ", shape(ds))
print("ds_train: ", shape(ds_train))
print("ds_test: ", shape(ds_test))


def batch_generator_train(batch_size, rnn_sequence_length):
    while True:
        X1_shape = (batch_size, rnn_sequence_length, 13)
        X1 = np.zeros(shape=X1_shape, dtype=np.float16)

        X2_shape = (batch_size, 13)
        X2 = np.zeros(shape=X2_shape, dtype=np.float16)

        Y1_shape = (batch_size, 1)
        Y1 = np.zeros(shape=Y1_shape, dtype=np.float16)

        Y2_shape = (batch_size, 1)
        Y2 = np.zeros(shape=Y2_shape, dtype=np.float16)

        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(len_train - (rnn_sequence_length + max_pred_step))
            # This points somewhere into the augmented series range.
            idaugmnt = np.random.randint(agmntCount)

            # Copy the sequences of data starting at this index.
            X1[i, :, :] = ds_train[idaugmnt, idx:idx + rnn_sequence_length, :]
            X2[i, :] = ds_train[idaugmnt, idx + rnn_sequence_length - 1, :]

            Y1[i, 0] = ds_train[idaugmnt, idx + rnn_sequence_length - 1, 0]
            Y2[i, 0] = ds_train[idaugmnt, idx + pred_step + rnn_sequence_length - 1, 6]
        yield [X1, X2], [Y1, Y2]


def batch_generator_validation(batch_size, rnn_sequence_length):
    while True:
        X1_shape = (batch_size, rnn_sequence_length, 13)
        X1 = np.zeros(shape=X1_shape, dtype=np.float16)

        X2_shape = (batch_size, 13)
        X2 = np.zeros(shape=X2_shape, dtype=np.float16)

        Y1_shape = (batch_size, 1)
        Y1 = np.zeros(shape=Y1_shape, dtype=np.float16)

        Y2_shape = (batch_size, 1)
        Y2 = np.zeros(shape=Y2_shape, dtype=np.float16)

        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(len_test - (rnn_sequence_length + max_pred_step))
            # This points somewhere into the augmented series range.
            idaugmnt = np.random.randint(agmntCount)

            # Copy the sequences of data starting at this index.
            X1[i, :, :] = ds_test[idaugmnt, idx:idx + rnn_sequence_length, :]
            X2[i, :] = ds_test[idaugmnt, idx + rnn_sequence_length - 1, :]

            Y1[i, 0] = ds_test[idaugmnt, idx + rnn_sequence_length, 0]
            Y2[i, 0] = ds_test[idaugmnt, idx + pred_step + rnn_sequence_length - 1, 6]
        yield [X1, X2], [Y1, Y2]


generator_traindata = batch_generator_train(batch_size=batch_size, rnn_sequence_length=rnn_sequence_length)
generator_validdata = batch_generator_validation(batch_size=batch_size, rnn_sequence_length=rnn_sequence_length)

# [tmpX1, tmpX2], [tmpY1, tmpY2] = next(generator_validdata)
# print(tmpX1[0, :, 0])
# print(tmpY1[0, :, :])

# ###################### Main #########################################

recurrent_input = Input(shape=(rnn_sequence_length, 13), name='recurrent_input')
rec_layer_1 = LSTM(74, return_sequences=True)(recurrent_input)
rec_layer_2 = LSTM(50)(rec_layer_1)
recurrent_output = Dense(1, activation='tanh', name='recurrent_output')(rec_layer_2)

sequential_input = Input(shape=(13,), name='sequential_input')
x = keras.layers.concatenate([rec_layer_2, sequential_input])

x = Dense(64, activation='tanh')(x)
x = Dense(64, activation='tanh')(x)
x = Dense(64, activation='tanh')(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)
model = Model(inputs=[recurrent_input, sequential_input], outputs=[recurrent_output, main_output])
plot_model(model, to_file='model.png', show_shapes=True)

model.compile(optimizer='Adam',
              loss={'main_output': 'binary_crossentropy', 'recurrent_output': 'logcosh'},
              loss_weights={'main_output': 1., 'recurrent_output': 0.50})

path_checkpoint = '../models/' + str(blc_id) + '_binary_on_' + str(pred_step) + '.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint, monitor='val_main_output_loss', verbose=1, save_weights_only=True, save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_main_output_loss', min_delta=1e-5, patience=10, verbose=1)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.1, min_lr=1e-5, patience=0, verbose=1)
callbacks = [callback_early_stopping, callback_checkpoint, callback_reduce_lr]

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

steps_per_epoch = int((len_train - rnn_sequence_length) * agmntCount / batch_size)
validation_steps = int((len_test - rnn_sequence_length) * agmntCount / batch_size)
print("steps_per_epoch = ", steps_per_epoch)
print("validation_steps = ", validation_steps)

model.fit_generator(generator=generator_traindata, epochs=1000000, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                    validation_data=generator_validdata, callbacks=callbacks)

try:
    model.load_weights(path_checkpoint)
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

f = open((str(blc_id) + '_binary_on_' + str(pred_step) + ".txt"), 'w+')
for check_index in range(0, 1):
    [tmpX1, tmpX2], [tmpY1, tmpY2] = next(generator_validdata)
    [pred1, pred2] = model.predict({'recurrent_input': tmpX1, 'sequential_input': tmpX2})
    print('true vs pred:', file=f)
    for line in range(0, batch_size):
        print(tmpY2[line, 0], pred2[line, 0], sep=',', file=f)