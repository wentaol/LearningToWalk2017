from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import *


# Model defined using functional API
def getShallowModel(env):
    nb_actions = env.get_action_dim()
    nb_obs = (env.get_observation_dim(), )
    # mu model
    mu_model_input = Input(shape=(1,) + nb_obs, 
                        name='mu_model_input')
    x = Flatten()(mu_model_input)
    x = Dense(400)(x) 
    x = LeakyReLU()(x)
    x = Reshape((2,200,1))(x)
    x = Conv2D(200, (1,200))(x) 
    x = LeakyReLU()(x)
    x = Conv2D(nb_actions/2, (1,1))(x)
    x = Reshape((nb_actions,))(x)
    x = Activation('tanh')(x)
    x = Activation('relu')(x)
    x = ActivityRegularization(l1=0.0001)(x)
    mu_model = Model(inputs=mu_model_input, outputs=x)

    # L model
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(
        shape=(1,) + nb_obs, name='observation_input')
    x = concatenate([action_input, Flatten()(observation_input)])
    x = Dense(400)(x) 
    x = LeakyReLU()(x)
    x = Dense(300)(x) 
    x = LeakyReLU()(x)
    x = Dense(1)(x)
    L_model = Model(input=[action_input, observation_input], output=x)
    return mu_model, L_model, action_input

