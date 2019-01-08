import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, CuDNNLSTM, Dropout, Lambda, RepeatVector, \
	Reshape, Permute, merge, Flatten

import keras.backend as K

def LSTM_model(look_back):
	model = Sequential()

	model.add(CuDNNLSTM(128, batch_input_shape=(1,look_back,1),stateful = True, return_sequences=True))
	model.add(Dropout(0.1))
	
	model.add(CuDNNLSTM(128,stateful = True, return_sequences=True))
	model.add(Dropout(0.1))

	model.add(CuDNNLSTM(128,stateful = True, return_sequences=True))
	model.add(Dropout(0.1))

	model.add(CuDNNLSTM(128,stateful = True))
	model.add(Dropout(0.1))
	
	model.add(Dense(3,activation='relu'))
	return model

def LSTM_model_not_stateful(look_back):
	model = Sequential()

	model.add(CuDNNLSTM(64, batch_input_shape=(1,look_back,1), return_sequences=True))
	model.add(Dropout(0.1))
	
	model.add(CuDNNLSTM(64, return_sequences=True))
	model.add(Dropout(0.1))

	model.add(CuDNNLSTM(64))
	model.add(Dropout(0.1))

	model.add(Dense(1, activation='sigmoid'))
	return model

# very bad
def simple_LSTM_model(look_back):
	model = Sequential()
	model.add(CuDNNLSTM(64, input_shape=(look_back,1)))
	model.add(Dense(1,activation='sigmoid'))

	return model

# um.. bad
def simple_LSTM_model_1(look_back):
	model = Sequential()
	model.add(CuDNNLSTM(64, input_shape=(look_back,1),return_sequences=True))
	model.add(CuDNNLSTM(64, return_sequences=True))
	model.add(CuDNNLSTM(64, return_sequences=True))
	model.add(CuDNNLSTM(64, return_sequences=True))
	model.add(CuDNNLSTM(64, return_sequences=True))
	model.add(CuDNNLSTM(64))
	model.add(Dense(1,activation='sigmoid'))

	return model

def simple_LSTM_model_2(look_back):
	model = Sequential()
	model.add(CuDNNLSTM(32, batch_input_shape=(1,look_back,1),return_sequences=True,stateful=True))
	model.add(CuDNNLSTM(32, return_sequences=True,stateful=True))
	model.add(CuDNNLSTM(32, return_sequences=True,stateful=True))
	model.add(CuDNNLSTM(32,stateful=True))
	model.add(Dense(1,activation='sigmoid'))

	return model


def Attention_LSTM_model(look_back):
	inputs = Input(shape=(look_back,1,))

	lstm_out = CuDNNLSTM(64, return_sequences=True)(inputs)
	lstm_out_dim = lstm_out.shape[2]

	a = Permute((2,1))(lstm_out)
	a = Reshape((int(lstm_out_dim), look_back))(a)
	a = Dense(look_back, activation='softmax')(a)

	a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
	a = RepeatVector(int(lstm_out_dim))(a)

	a_probs = Permute((2,1), name='attention_vec')(a)
	attention_mul = merge([lstm_out, a_probs], name='attention_mul', mode='mul')

	attention_mul = Flatten()(attention_mul)


	#dense = Dense(64, activation='tanh')(attention_mul)
	output = Dense(1, activation='sigmoid')(attention_mul)

	model = Model(input=[inputs], output=output)

	return model

######################################################################
def attention_3d_block(inputs, look_back):
	# inputs.shape = (batch_size, time_steps, input_dim)
	input_dim = int(inputs.shape[2])
	a = Permute((2, 1))(inputs)
	a = Reshape((input_dim, look_back))(a) # this line is not useful. It's just to know which dimension is what.
	a = Dense(look_back, activation='softmax')(a)
	
	a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
	a = RepeatVector(input_dim)(a)
	a_probs = Permute((2, 1), name='attention_vec')(a)
	output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
	return output_attention_mul


def model_attention_applied_after_lstm(look_back):
	inputs = Input(shape=(look_back, 1,))
	lstm_units = 64
	lstm_out = CuDNNLSTM(lstm_units, return_sequences=True)(inputs)
	attention_mul = attention_3d_block(lstm_out, look_back)
	attention_mul = Flatten()(attention_mul)
	output = Dense(1, activation='sigmoid')(attention_mul)
	model = Model(input=[inputs], output=output)
	return model


def model_attention_applied_before_lstm(look_back):
	inputs = Input(shape=(look_back, 1,))
	attention_mul = attention_3d_block(inputs, look_back)
	lstm_units = 32
	attention_mul = CuDNNLSTM(lstm_units, return_sequences=False)(attention_mul)
	output = Dense(1, activation='sigmoid')(attention_mul)
	model = Model(input=[inputs], output=output)
	return model
######################################################################

def ANN_model(look_back):
	model = Sequential()
	model.add(Flatten(input_shape=(look_back,1)))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(3, activation='relu'))

	return model