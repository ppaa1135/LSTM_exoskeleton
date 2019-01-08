
from pandas import DataFrame, read_csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy.signal import savgol_filter 

import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Dropout, Lambda, RepeatVector, \
	Reshape, Permute, merge, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
#from keras.utils.training_utils import multi_gpu_model

import keras.backend as K

from make_data_set import *
from lstm_model import *


def create_dataset(dataset, look_back):

	#input : dataset(multi array), look_back(how many point to predict)
	#output : dataX, dataY
	
	dataX, dataY = [],[]

	for one_data in dataset:
		for i in range(len(one_data)-look_back-2):
			dataX.append([list(one_data[i+j]) for j in range(look_back)])
			dataY.append([one_data[i+look_back][0],one_data[i+look_back+1][0],one_data[i+look_back+2][0]])

	return np.array(dataX), np.array(dataY)



def save_model_and_fit(model, model_file_name, x_train, y_train, x_test, y_test):
	
	
	sgd = keras.optimizers.SGD(lr=0.001, momentum=0.1, nesterov = False)
	adam = keras.optimizers.Adam(lr=0.0001)

	## multi gpu
	#model = multi_gpu_model(model, gpus=2)

	#model.compile(loss='mean_squared_error', optimizer=adam)
	model.compile(loss='mean_absolute_error', optimizer=adam)
	#model.compile(loss='mean_absolute_error', optimizer='RMSprop')
	#tb_hist = keras.callbacks.TensorBoard(log_dir='graph', histogram_freq=0, write_graph=True,write_images=True)
	try:
		model.fit(x_train, y_train, epochs=100, batch_size=1, validation_data=(x_test, y_test))
		model.save_weights('save_model/%s' %MODEL_FILE_NAME)
	except KeyboardInterrupt:
		model.save_weights('save_model/%s' %MODEL_FILE_NAME)


def load_model_and_fit(model, model_file_name, x_train, y_train, x_test, y_test):
	model.load_weights('save_model/%s' %model_file_name)

	sgd = keras.optimizers.SGD(lr=0.001, momentum=0.1, nesterov = False)
	adam = keras.optimizers.Adam(lr=0.00001)

	#model.compile(loss='mean_squared_error', optimizer=adam)
	#model.compile(loss='mean_absolute_error', optimizer=adam)
	model.compile(loss='mean_absolute_error', optimizer=adam)
	
	try:
		model.fit(x_train, y_train, epochs=50, batch_size=1, validation_data=(x_test, y_test))
		model.save_weights('save_model/%s' %MODEL_FILE_NAME)
	except KeyboardInterrupt:
		model.save_weights('save_model/%s' %MODEL_FILE_NAME)
	


def load_model_and_predict_and_plot_save(model,model_file_name,x_test,y_test,x_train,y_train,data_size):
	model.load_weights('save_model/%s' %model_file_name)

	evaluate_num = 0
	error_list = np.zeros((20))

	for j in range(int(data_size*0.25)):

		xhat = x_test[j*(one_data_size-look_back-2)]

		predictions = np.zeros((look_ahead,3))
		plt_predictions = np.zeros((look_ahead,1))
		plt_y_test = np.zeros((look_ahead,1))
		

		for i in range(look_ahead):
			prediction = model.predict(np.array([xhat]), batch_size=1) #1,40,1

			predictions[i] = prediction # + diff

			if i == 0:
				xhat = np.vstack([xhat[1:],[prediction[0][0]]])
				plt_predictions[i] = prediction[0][0]

			elif i == 1:
				xhat = np.vstack([xhat[1:],[(prediction[0][0] + predictions[i-1][1])/2]])
				plt_predictions[i] = (prediction[0][0] + predictions[i-1][1])/2

			else:
				xhat = np.vstack([xhat[1:],[(prediction[0][0] + predictions[i-2][2] + predictions[i-1][1])/3]])
				plt_predictions[i] = (prediction[0][0] + predictions[i-2][2] + predictions[i-1][1])/3

			

		for i in range(look_ahead-2):
			plt_y_test[i] = y_test[(look_ahead-2)*j+i][0]
		plt_y_test[-1] = y_test[(look_ahead-2)*(j+1)-1][2]
		plt_y_test[-2] = y_test[(look_ahead-2)*(j+1)-1][1]

		plt_y_test = plt_y_test.flatten()
		#print(plt_y_test)
		'''
		tmp_y_test = y_test[look_ahead*j:look_ahead*(j+1)]
		tmp_y_test = tmp_y_test.reshape(-1,1)

		predictions = scaler.inverse_transform(predictions)
		original = scaler.inverse_transform(tmp_y_test)
		tmp_y_test = tmp_y_test.reshape(tmp_y_test.shape[0])
		'''

		############# plot test figure
		plt.figure()
		plt.ylim(-30,60)
		
		############
		plt.plot(plt_y_test-30,'b-',label="original")
		#plt.plot(original,'b-',label="original")
		plt.plot(plt_predictions-30,'r-',label="predict")
		
		plt.legend(loc="upper left")
		plt.ylabel("LHipAngle")
		plt.xlabel("Time")

		plt.savefig("save_fig/LHipAngle (1) predict/test/%d" %j)
		plt.close()



		############# plot error figure
		plt.figure()
		plt.ylim(0,30)

		################ error y_test---61,3  plt_predictions---61,1
		plt.bar(np.arange(0,look_ahead,1),abs(plt_y_test-plt_predictions.reshape(plt_predictions.shape[0])))

		plt.xlabel("Absolute Error")
		plt.ylabel("Time")

		plt.savefig("save_fig/LHipAngle (1) error/test/%d" %j)
		plt.close()
		#print("save! : ",j)
		
		print(mean_absolute_error(plt_y_test,plt_predictions))
		if mean_absolute_error(plt_y_test,plt_predictions) <= 5.0:
			evaluate_num += 1

		error_list[int(mean_absolute_error(plt_y_test,plt_predictions))] += 1

	x_label = []
	for i in range(20):
		x_label.append("%s~%s" %(i,i+1))

	plt.figure()
	plt.bar(np.arange(0,20,1),error_list)
	plt.savefig("save_fig/error")
	plt.close()

	print("total : %d / %d, percentage : %f" %(evaluate_num, int(data_size*0.25), float(evaluate_num/(data_size*0.25)*100)))
	print(error_list)


################################ Parameter ####################################

batch_size = 1

one_data_size = 101
look_back = 40
look_ahead = one_data_size - look_back

#MODEL_FILE_NAME = 'LSTM_model_not_stateful'
MODEL_FILE_NAME = 'ANN_model2'
LABEL_NAME = 'LHipAngles (1)'
DATASET_NAME = 'LHipAngles (1) generation'
###############################################################################

if __name__ =='__main__':

	# excel to pickle write
	#dataset = one_label_dataset(LABEL_NAME)
	#pickle_file_write(dataset, LABEL_NAME)

	# excel pickle read
	#dataset = pickle_file_read(LABEL_NAME) #324 101
	#dataset = np.array(dataset)

	# gait dataset & data generation 3 times
	#dataset = dataset[:108][:]
	#dataset = data_generation(dataset)

	# dataset to pickle write & read
	#pickle_file_write(dataset, DATASET_NAME)
	dataset = pickle_file_read(DATASET_NAME)

	# flatten
	data_size = dataset.shape[0]
	dataset = dataset.flatten() # 324*101

	# [1]Gait #################################
	#dataset = dataset[:one_data_size*108]
	#dataset = savgol_filter(dataset, 51, 3)

	# (0,1) nomalization
	dataset = dataset.reshape(-1,1)
	dataset = dataset + 30
	#scaler = MinMaxScaler(feature_range = (0, 1))
	#dataset = scaler.fit_transform(dataset)

	# test / train
	dataset_train = []; dataset_test = []
	cnt = 0
	for i in range(data_size):
		if cnt % 4 == 0:
			dataset_test.append(dataset[i*one_data_size:(i+1)*one_data_size])
		else:
			dataset_train.append(dataset[i*one_data_size:(i+1)*one_data_size])
		cnt += 1

	dataset_train = np.array(dataset_train)
	dataset_test = np.array(dataset_test)

	
	x_train, y_train = create_dataset(dataset_train, look_back) 
	x_test, y_test = create_dataset(dataset_test, look_back) 

	#x_train = x_train.reshape(x_train.shape[0],x_train.shape[1])
	#x_test = x_test.reshape(x_test.shape[0],x_test.shape[1])

	#y_train = y_train.flatten()
	#y_test = y_test.flatten()

	model = ANN_model(look_back)
	#model = simple_LSTM_model(look_back)
	#model = LSTM_model(look_back)
	#model = LSTM_model_not_stateful(look_back)
	#model = Attention_LSTM_model(look_back)
	#model = model_attention_applied_after_lstm(look_back)

	save_model_and_fit(model, MODEL_FILE_NAME, x_train, y_train, x_test, y_test)
	#load_model_and_fit(model, MODEL_FILE_NAME, x_train, y_train, x_test, y_test)
	load_model_and_predict_and_plot_save(model, MODEL_FILE_NAME ,x_test,y_test,x_train,y_train,data_size)
	# list -> numpy

	'''
	dataset_train = np.array(dataset_train)
	dataset_test = np.array(dataset_test)
	
	print(dataset_train.shape)

	# ================================================
	x_train, y_train = create_dataset(np.array(dataset_train), look_back) # 4700,100
	x_test, y_test = create_dataset(np.array(dataset_test), look_back) # 1180, 20

	print(x_train.shape)

	y_train = y_train.flatten()
	y_test = y_test.flatten()

	#model = simple_LSTM_model_2(look_back)
	model = LSTM_model(look_back)
	#model = Attention_LSTM_model(look_back)
	
	save_model_and_fit(model, MODEL_FILE_NAME, x_train, y_train, x_test, y_test)
	'''
	#load_model_and_fit(model, MODEL_FILE_NAME, x_train, y_train, x_test, y_test)
	#load_model_and_predict(model, MODEL_FILE_NAME ,x_test,y_test)
