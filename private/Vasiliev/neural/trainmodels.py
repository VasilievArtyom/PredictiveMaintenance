import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Embedding, Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

plt.rc('text', usetex=True)

outpath = "../../../plots/temperatures"
inpath = "../../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
                        ('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)


max_possible_model_temperature = 15.0
min_possible_model_temperature = 70.0
def scale_T(_T):
    return (_T - min_possible_model_temperature) / (max_possible_model_temperature - min_possible_model_temperature)

def unscale_T(_T):
    return (_T) * (max_possible_model_temperature - min_possible_model_temperature) + min_possible_model_temperature

sT = scale_T(T)

print("Min:", np.min(sT))
print("Max:", np.max(sT))

delta = np.amin(np.abs(sT[:-2, :] - sT[1:-1, :]))

#data augmintation -- 50000 slightly tuned copies of T
agmntCount=50000
agmntdT=np.zeros((agmntCount, np.size(sT[:,0]), np.size(sT[0,:])))
agmntdT[0,:,:] = sT
np.random.seed(0)
mu, sigma = 0, delta*100
for i in range(1, agmntCount):
    agmntdT[i] = agmntdT[0] + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))

predictSteps = 1
#Keep dataset tail for validate prediction quality
cutFromTail = 60
#In order to have shifted and unshifted series with same shape
t = cutFromTail + predictSteps

inData = np.zeros((agmntCount, np.size(sT[:-t,0]), np.size(sT[0,:]) + 1))
for i in range(0, agmntCount):
    inData[i,:,0] = Mode[:-t] / np.amax(Mode[:-t])
    inData[i,:,1:]= agmntdT[i,:-t,:]

num_data = len(inData[0,:,0])
train_split = 0.80
num_train = int(train_split * num_data)
num_test = num_data - num_train
inData_train = inData[:, 0:num_train, :]
inData_test = inData[:, num_train:, :]
num_inData_signals = inData.shape[2]


for outputBlockId in range(0, 10):
    outData = np.expand_dims(agmntdT[:,predictSteps:-cutFromTail,outputBlockId], axis=2)
    outData_train = outData[:, 0:num_train, :]
    outData_test = outData[:, num_train:, :]
    num_outData_signals = outData.shape[2]
    
    def batch_generator(batch_size, sequence_length):
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
                in_batch[i] = inData_train[idaugmnt, idx:idx+sequence_length,:]
                out_batch[i] = outData_train[idaugmnt, idx:idx+sequence_length,:]
        
        yield (in_batch, out_batch)
    
    batch_size = 128
    sequence_length=512
    
    generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
    
    in_batch, out_batch = next(generator)
    validation_data = (np.expand_dims(inData_test[0,:,:], axis=0), np.expand_dims(outData_test[0,:,:], axis=0))
    
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(None, num_inData_signals,)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(10, return_sequences=True))
    model.add(LSTM(3, return_sequences=True))
    model.add(Dense(1, activation = "linear"))
    model.compile(Adam(learning_rate=1e-3), loss='mean_absolute_error')
    model.summary()
    
    path_checkpoint = str(outputBlockId)+'_checkpoint_stacked_LSTM.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=False,
                                      save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=5, verbose=1)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=1)
    callbacks = [callback_early_stopping,
            callback_checkpoint,
            callback_reduce_lr]
    model.fit_generator(generator=generator,
                        epochs=1000,
                        steps_per_epoch=50,
                        validation_data=validation_data,
                        callbacks=callbacks)
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)
    result = model.evaluate(x=np.expand_dims(inData_test[0], axis=0),
                        y=np.expand_dims(outData_test[0], axis=0))
    print("loss (test-set):", result)