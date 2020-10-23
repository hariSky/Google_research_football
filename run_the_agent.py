import gfootball.env as football_env
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import load_model


import tensorflow as tf
tf.compat.v1.disable_eager_execution()


##  If you wish to just perform inference with your model and not further optimization or training your model, you can simply wish to ignore the loss function like this:
model_actor = load_model("model_actor_10.hdf5", compile=False)

####model = load_model("model_actor_0.hdf5", custom_objects={'loss': ppo_loss()})



image_based = False

if image_based:
    env = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
else:
    env = football_env.create_environment(env_name='academy_empty_goal', representation='simple115')


state = env.reset()
state_dims = env.observation_space.shape
n_actions = env.action_space.n


env_display = football_env.create_environment(env_name='academy_empty_goal', representation='pixels', render=True)
state_display = env_display.reset()
state_dims_display = env_display.observation_space.shape
n_actions_display = env_display.action_space.n

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

tensor_board = TensorBoard(log_dir='./logs')



ppo_steps = 128
target_reached = False
best_reward = 0
iters = 0
max_iters = 100000

while not target_reached and iters < max_iters:

    states = []
    actions = []
    values = []
    masks = []
    rewards = []
    actions_probs = []
    actions_onehot = []
    state_input = None

    for itr in range(ppo_steps):
        state_input = K.expand_dims(state, 0)
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.random.choice(n_actions, p=action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1

        observation_display, reward_display, done_display, info_display = env_display.step(action)
        observation, reward, done, info = env.step(action)
        state = observation
        
        #print("done : {}".format(done))
        #print("done_display : {}".format(done_display))  
        
        if done:
            #print("IN done")
            env_display.reset()        
            env.reset()  
        
        if done_display:
            #print("IN done_display")
            env_display.reset()        
            env.reset()          
        
        
        
        
        
        
        
        
        
        
        
        
        
