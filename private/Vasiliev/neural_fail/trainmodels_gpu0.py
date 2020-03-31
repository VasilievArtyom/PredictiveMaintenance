import math
import numpy as np
from numpy import *
from scipy.optimize import curve_fit
from scipy import signal
from os import path
import os
import matplotlib.pyplot as plt


from keras import backend as K

from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Embedding, Dense, Activation, Dropout, LSTM
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.python.client import device_lib


plt.rc('text', usetex=True)

outpath = ""
inpath = "../../../"

currentfile = "Imitator_2_2400.csv"

# Read from file
strdatatype = np.dtype([('N', np.int_, (2,)), ('Time_Count', np.int_ ), ('Mode', np.int_ ),
                        ('T', np.float_, (10,)), ('S', np.bool_, (10,)), ('System_State', np.bool_ )])
N, Time_Count, Mode, T, S, System_State = np.loadtxt(path.join(inpath, currentfile), 
                                                     unpack=True, delimiter=';', skiprows=1, dtype=strdatatype)

max_possible_model_temperature = 18
min_possible_model_temperature = 66
def scale_T(_T):
    return ((_T - min_possible_model_temperature) / (max_possible_model_temperature - min_possible_model_temperature)) * 2.0 - 1.0

def unscale_T(_T):
    return ((_T + 1.0) / 2.0) * (max_possible_model_temperature - min_possible_model_temperature) + min_possible_model_temperature

warmup_steps = 10
def loss_mse_warmup(y_true, y_pred): 
	return K.mean(K.square(y_pred[:, warmup_steps:, :]-y_true[:, warmup_steps:, :]))

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
	fig, ax = plt.subplots(figsize=(15,5))
    
	# Plot and compare the two signals.
	ax.plot(signal_true, label='true')
	ax.scatter(np.arange(0, len(signal_pred[0])), signal_pred[0,:,:], label='pred', color='r')

	# Plot labels etc.
	plt.legend()
	plt.tight_layout()
	plt.draw()
	fig.savefig(path.join(outpath, name))
	plt.clf()

sT = scale_T(T)
print(np.amax(T), np.amin(T))
print(np.amax(sT), np.amin(sT))
delta = np.amin(np.abs(sT[:-2, :] - sT[1:-1, :])) 

#data augmintation -- 50000 slightly tuned copies of T
agmntCount=5000
agmntdT=np.zeros((agmntCount, np.size(sT[:,0]), np.size(sT[0,:])))
agmntdT[0,:,:] = sT
np.random.seed(0)
mu, sigma = 0, delta * 500
for i in range(1, agmntCount):
	agmntdT[i] = agmntdT[0] + np.random.normal(mu, sigma, (np.size(sT[:,0]), np.size(sT[0,:])))


predictSteps = 20
sequence_length=200

#Keep dataset tail for validate prediction quality
cutFromTail = 150

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print(device_lib.list_local_devices())
for outputBlockId in range(0, 5):
	for _predictSteps in range(1, predictSteps + 1):
		#In order to have shifted and unshifted series with same shape
		t = cutFromTail + predictSteps
		inData = np.zeros((agmntCount, np.size(sT[:-t,0]), np.size(sT[0,:])))
		for i in range(0, agmntCount):
			inData[i,:,:]= agmntdT[i,:-t,:]
		outData = np.expand_dims(agmntdT[:,predictSteps:-cutFromTail,outputBlockId], axis=2) 

		num_data = len(inData[0,:,0])
		train_split = 0.80
		num_train = int(train_split * num_data)
		num_test = num_data - num_train
		
		inData_train = inData[:, 0:num_train, :]
		inData_test = inData[:, num_train:, :]

		outData_train = outData[:, 0:num_train, :]
		outData_test = outData[:, num_train:, :]

		num_inData_signals = inData.shape[2]
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

		batch_size = 512
		generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
		in_batch, out_batch = next(generator)

		validation_data = (np.expand_dims(inData_test[0,:,:], axis=0), np.expand_dims(outData_test[0,:,:], axis=0))

		model = Sequential()
		model.add(LSTM(50, return_sequences=True, input_shape=(None, num_inData_signals,)))
		model.add(Dense(20, activation = "linear"))
		model.add(LSTM(50, return_sequences=True))
		model.add(Dense(1, activation = "linear"))

		model.compile(Adam(learning_rate=1e-3), loss=loss_mse_warmup)

		path_checkpoint = 'model/'+str(outputBlockId)+'_multistep_on_'+str(predictSteps)+'_steps.keras'
		callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
												monitor='val_loss',
												verbose=1,
												save_weights_only=True,
												save_best_only=True)
		callback_early_stopping = EarlyStopping(monitor='val_loss', patience=25, verbose=1)
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

		model.fit_generator(generator=generator, epochs=1000000, steps_per_epoch=350, validation_data=validation_data, callbacks=callbacks)

		try:
			model.load_weights(path_checkpoint)
		except Exception as error:
			print("Error trying to load checkpoint.")
			print(error)

		plot_comparison(0, str(outputBlockId)+'_multistep_on_'+str(_predictSteps), length=400, train=False)
		inData = None
		inData_test = None
		inData_train = None
		outData = None
		outData_test = None
		outData_train = None
		del inData
		del inData_test
		del inData_train
		del outData
		del outData_test
		del outData_train