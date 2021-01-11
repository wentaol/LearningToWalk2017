from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.merge import Concatenate


def getNAFAModel(WINDOW_LENGTH, env):
	nb_actions = env.action_space.shape[0]
	# Create networks
	V_model = Sequential()
	V_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
	V_model.add(Dense(16))
	V_model.add(Activation('relu'))
	V_model.add(Dense(1))
	V_model.add(Activation('linear'))

	mu_model = Sequential()
	mu_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
	mu_model.add(Dense(16))
	mu_model.add(Activation('relu'))
	mu_model.add(Dense(nb_actions))
	mu_model.add(Activation('linear'))

	action_input = Input(shape=(nb_actions,), name='action_input')
	observation_input = Input(
	    shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')
	x = concatenate([action_input, Flatten()(observation_input)])
	x = Dense(16)(x)
	x = Activation('relu')(x)
	x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
	x = Activation('linear')(x)
	L_model = Model(input=[action_input, observation_input], output=x)
	return V_model, mu_model, L_model

# Model defined using functional API
def getSymModel(WINDOW_LENGTH, env):
	nb_actions = env.action_space.shape[0]
	# Build all necessary models: V, mu, and L networks.
	# V model
	V_model_input = Input(shape=(1,) + env.observation_space.shape, 
	                    name='V_model_inpu')
	x = Flatten()(V_model_input)
	x = Dense(32)(x)
	x = Activation('relu')(x)
	x = Dense(1)(x)
	x = Activation('linear')(x)
	V_model = Model(inputs=V_model_input, outputs=x)
	# mu model
	mu_model_input = Input(shape=(1,) + env.observation_space.shape, 
	                    name='mu_model_input')
	x = Flatten()(mu_model_input)
	x = Dense(16)(x)
	x = Activation('relu')(x)
	x = Reshape((2,8,1))(x)
	print x.shape
	x = Conv2D(nb_actions/2, (1,8))(x)
	print x.shape
	x = Reshape((nb_actions,))(x)
	x = Activation('linear')(x)
	mu_model = Model(inputs=mu_model_input, outputs=x)
	# L modelexit
	action_input = Input(shape=(nb_actions,), name='action_input')
	observation_input = Input(
	    shape=(1,) + env.observation_space.shape, name='observation_input')
	x = concatenate([action_input, Flatten()(observation_input)])
	x = Dense(16)(x)
	x = Activation('relu')(x)
	x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
	x = Activation('linear')(x)
	L_model = Model(input=[action_input, observation_input], output=x)
	return V_model, mu_model, L_model