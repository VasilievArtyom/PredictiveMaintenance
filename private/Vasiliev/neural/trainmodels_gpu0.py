import math
import numpy as np
from numpy import *
from os import path
import os
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import losses
from tensorflow.python.client import device_lib

plt.rc('text', usetex=True)

# <--------------------->
# Tunable
predictSteps = 20
# <--------------------->

sequence_length_in = predictSteps * 4
sequence_length_out = predictSteps
sequence_length = sequence_length_in + sequence_length_out
# Keep dataset tail for validate prediction quality
cutFromTail = 60

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# <--------------------->
# Tunable
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
batch_size = 125
# <--------------------->
print(device_lib.list_local_devices())


def plot_comparison(start_idx, name, length=100, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = inData_train[0]
        y_true = outData_train[0]
    else:
        # Use test-data.
        x = inData_test[0]
        y_true = outData_test[0]
    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)
    # Use the model to predict the output-signals.
    y_pred = model.predict(x)
    print(shape(x))
    print(shape(y_pred))

    # Get the output-signal predicted by the model.
    signal_pred = y_pred

    # Get the true output-signal from the data-set.
    signal_true = y_true

    # Make the plotting-canvas bigger.
    fig, ax = plt.subplots(figsize=(15, 5))

    # Plot and compare the two signals.
    ax.plot(signal_true, label='true')
    ax.scatter(np.arange(
        0, len(signal_pred[0])), signal_pred[0, :, :], label='pred', color='r')

    # Plot labels etc.
    plt.legend()
    plt.tight_layout()
    plt.draw()
    fig.savefig(path.join(outpath, name))
    plt.clf()


# <--------------------->
# Tunable
warmup_steps = 0
importance_factor = 2.0
# <--------------------->


def loss_mse_weighted(y_true, y_pred):
    # weights = K.ones_like(y_true[:, :, :])
    # weights[:, :warmup_steps, :] = weights[:, :warmup_steps, :] / importance_factor
    # weights[:, -predictSteps:, :] = weights[:, -predictSteps:, :] * importance_factor
    # return K.mean(K.square((y_pred[:, :, :] - y_true[:, :, :]) * weights))
    # return K.mean(K.square(y_pred[:, warmup_steps:, :] - y_true[:, warmup_steps:, :])) + importance_factor * K.mean(K.square(y_pred[:, -predictSteps:, :] - y_true[:, -predictSteps:, :]))
    return K.mean(K.square(y_pred[:, warmup_steps:, :] - y_true[:, warmup_steps:, :])) + importance_factor * K.mean(K.square(y_pred[:, -predictSteps:, :] - y_true[:, -predictSteps:, :]))


def read_dtaset_by_index(index):
    inpath = "data/"
    currentfile = path.join(inpath, "data_T_{0}.csv".format(index))
    # Read from file
    strdatatype = np.dtype([('N', np.int_), ('Mode', np.float_),
                            ('T', np.float_, (10,)),
                            ('kalmanT', np.float_, (10,)),
                            ('kalmanT_dot', np.float_, (10,)),
                            ('ma3T', np.float_, (10,)),
                            ('ma13T', np.float_, (10,)),
                            ('ma55T', np.float_, (10,)),
                            ('ma144T', np.float_, (10,))])
    # N, _Mode, _T, _kalmanT, _ma2T, _ma3T, _ma5T, _ma8T, _ma13T
    return np.loadtxt(currentfile, unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


# Read unaugmented dataset
N, Mode, T, kalmanT, kalmanT_dot, ma3T, ma13T, ma55T, ma144T = read_dtaset_by_index(0)
# Read agmntCount augmented copies and collect full dataset
agmntCount = 200
# In order to have shifted and unsifted series with same shape
t = cutFromTail + predictSteps

dataset = np.empty((agmntCount, len(T[144:-t, 0]), 61))
shifted_dataset = np.empty((agmntCount, len(T[144:-t, 0]), 61))
for agmnt_index in range(0, agmntCount):
    _N, _Mode, _T, _kalmanT, _kalmanT_dot, _ma3T, _ma13T, _ma55T, _ma144T = read_dtaset_by_index(agmnt_index)
    dataset[agmnt_index, :, 60] = _Mode[144:-t]
    dataset[agmnt_index, :, 0:10] = _kalmanT[144:-t, :]
    dataset[agmnt_index, :, 10:20] = _kalmanT_dot[144:-t, :]
    dataset[agmnt_index, :, 20:30] = _ma3T[144:-t, :]
    dataset[agmnt_index, :, 30:40] = _ma13T[144:-t, :]
    dataset[agmnt_index, :, 40:50] = _ma55T[144:-t, :]
    dataset[agmnt_index, :, 50:60] = _ma144T[144:-t, :]

    shifted_dataset[agmnt_index, :, 60] = _Mode[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 0:10] = _kalmanT[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 10:20] = _kalmanT_dot[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 20:30] = _ma3T[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 30:40] = _ma13T[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 40:50] = _ma55T[144 + predictSteps:-cutFromTail]
    shifted_dataset[agmnt_index, :, 50:60] = _ma144T[144 + predictSteps:-cutFromTail]

num_data = len(dataset[0, :, 0])
train_split = 0.85
num_train = int(train_split * num_data)
num_test = num_data - num_train

print("num_train = ", num_train)
print("num_test = ", num_test)

inData_train = dataset[:, 0:num_train, :]
inData_test = dataset[:, num_train:, :]


num_inData_signals = inData_train.shape[2]
num_outData_signals = 1


def batch_generator_train(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        in_shape = (batch_size, sequence_length, num_inData_signals)
        in_batch = np.zeros(shape=in_shape, dtype=np.float16)
        # Allocate a new array for the batch of output-signals.
        out_shape = (batch_size, sequence_length, num_outData_signals)
        out_batch = np.zeros(shape=out_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)
            # This points somewhere into the augmented series range.
            idaugmnt = np.random.randint(agmntCount)

            # Copy the sequences of data starting at this index.
            in_batch[i] = inData_train[idaugmnt, idx:idx + sequence_length, :]
            out_batch[i] = outData_train[idaugmnt, idx:idx + sequence_length, :]
        yield (in_batch, out_batch)


def batch_generator_validation(batch_size, sequence_length):
    """
    Generator function for creating random batches of validation-data.
    """
    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        in_shape = (batch_size, sequence_length, num_inData_signals)
        in_batch = np.zeros(shape=in_shape, dtype=np.float16)
        # Allocate a new array for the batch of output-signals.
        out_shape = (batch_size, sequence_length, num_outData_signals)
        out_batch = np.zeros(shape=out_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_test - sequence_length)
            # This points somewhere into the augmented series range.
            idaugmnt = np.random.randint(agmntCount)

            # Copy the sequences of data starting at this index.
            in_batch[i] = inData_test[idaugmnt, idx:idx + sequence_length, :]
            out_batch[i] = outData_test[idaugmnt, idx:idx + sequence_length, :]
        yield (in_batch, out_batch)


# ###################### Main #########################################
for outputBlockId in range(0, 5):
    outData_train = np.expand_dims(shifted_dataset[:, 0:num_train, outputBlockId], axis=2)
    outData_test = np.expand_dims(shifted_dataset[:, num_train:, outputBlockId], axis=2)

    generator_traindata = batch_generator_train(batch_size=batch_size, sequence_length=sequence_length)
    generator_validdata = batch_generator_validation(batch_size=batch_size, sequence_length=sequence_length)

    model = Sequential()
    model.add(LSTM(1024, return_sequences=True, input_shape=(None, num_inData_signals,)))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dense(10, activation="linear"))
    # model.add(Dense(32, activation="tanh"))
    model.add(Dense(1, activation="tanh"))

    # model.compile(Adam(learning_rate=1e-3), loss=loss_mse_weighted)
    #model.compile(Adam(learning_rate=5e-3), loss=losses.logcosh)
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd, loss=losses.logcosh)

    path_checkpoint = 'models/' + str(outputBlockId) + '_multistep_on_' + str(predictSteps) + '_steps.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-5,
                                           patience=0,
                                           verbose=1)
    callbacks = [callback_early_stopping, callback_checkpoint, callback_reduce_lr]

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    model.fit_generator(generator=generator_traindata, epochs=1000000, steps_per_epoch=2713, validation_steps=321,
                        validation_data=generator_validdata, callbacks=callbacks)

    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

    outpath = ""
    plot_comparison(0, str(outputBlockId) + '_multistep_on_' + str(predictSteps), length=sequence_length, train=False)
