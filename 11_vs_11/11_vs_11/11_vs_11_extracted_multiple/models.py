import gfootball.env as football_env
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False



import logging
tf.get_logger().setLevel(logging.ERROR)
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
#import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)

clipping_val = 0.2
critic_discount = 0.5
#entropy_beta = 0.001
entropy_beta = 0.1


import tensorflow as tf
tf.compat.v1.disable_eager_execution()





def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        #total_loss = actor_loss - entropy_beta * K.mean(
            #-(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss





def get_model_actor_image(input_dims, output_dims, n_actions):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss( 
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])

    model.summary()
    return model


def get_model_actor_extracted(input_dims, output_dims, n_actions):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))


    x = Conv2D(512, kernel_size=(16, 16),padding="valid", kernel_initializer="glorot_uniform",bias_initializer='zeros',  activation='relu', input_shape=input_dims, name='1Conv', strides=(8,8))(state_input)
    #x = BatchNormalization(axis=3)(x)
    x = Conv2D(128, kernel_size=(8, 11),  kernel_initializer="glorot_uniform", bias_initializer='zeros', activation='relu', name='2Conv', strides=(2,2))(x)


    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)
    

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss( 
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    model.summary()
    return model





def get_model_actor_simple(input_dims, output_dims, n_actions):
    state_input = Input(shape=input_dims)
    oldpolicy_probs = Input(shape=(1, output_dims,))
    advantages = Input(shape=(1, 1,))
    rewards = Input(shape=(1, 1,))
    values = Input(shape=(1, 1,))

    # Classification block
    x = Dense(1024, activation='relu', name='fc1')(state_input)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    out_actions = Dense(n_actions, activation='softmax', name='predictions')(x)

    model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                  outputs=[out_actions])

               
    model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
        oldpolicy_probs=oldpolicy_probs,
        advantages=advantages,
        rewards=rewards,
        values=values)])
    # model.summary()
    return model


def get_model_critic_image(input_dims):
    state_input = Input(shape=input_dims)

    feature_extractor = MobileNetV2(include_top=False, weights='imagenet')

    for layer in feature_extractor.layers:
        layer.trainable = False

    # Classification block
    x = Flatten(name='flatten')(feature_extractor(state_input))
    x = Dense(1024, activation='relu', name='fc1')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model

    state_input = Input(shape=input_dims)


def get_model_critic_extracted(input_dims):
    state_input = Input(shape=input_dims)



    x = Conv2D(512, kernel_size=(16, 16),padding="valid", kernel_initializer="glorot_uniform",bias_initializer='zeros',  activation='relu', input_shape=input_dims, name='1Conv', strides=(8,8))(state_input)
    #x = BatchNormalization(axis=3)(x)
    x = Conv2D(128, kernel_size=(8, 11),  kernel_initializer="glorot_uniform", bias_initializer='zeros', activation='relu', name='2Conv', strides=(2,2))(x)

    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='dens1')(x)
    out_actions = Dense(1, activation='tanh', name='predictions')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    model.summary()
    return model

    state_input = Input(shape=input_dims)




def get_model_critic_simple(input_dims):
    state_input = Input(shape=input_dims)

    # Classification block
    x = Dense(1024, activation='relu', name='fc1')(state_input)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dense(32, activation='relu', name='fc3')(x)
    out_actions = Dense(1, activation='tanh')(x)

    model = Model(inputs=[state_input], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=1e-4), loss='mse')
    # model.summary()
    return model
