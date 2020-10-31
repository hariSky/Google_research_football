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

import tensorflow as tf
tf.compat.v1.disable_eager_execution()


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

gamma = 0.99
lmbda = 0.95

from models import get_model_actor_image,get_model_actor_extracted,get_model_actor_simple,get_model_critic_image,get_model_critic_extracted,get_model_critic_simple


z=np.zeros((72, 96))

z[30:42,0:2]=-0.5
z[30:42,2:4]=-0.46
z[30:42,4:7]=-0.44

z[0:20,0:7]=-0.4
z[20:30,0:7]=-0.42

z[42:52,0:7]=-0.42
z[52:72,0:7]=-0.40


# zero until the middle
c=0
for i in range(5, 48):
    z[:,i]=c*0.4/41 - 0.4
    c=c+1
    

# until 76
c=48
for i in range(48, 76):
    z[:,i]=c*0.4/41 - 0.4
    c=c+1
      
# from 76 middle force
c=76
for i in range(76, 96):
    z[30:42,i]=c*0.46/41 - 0.4
    c=c+1   
    
# from 76 uper less
c=76
for i in range(76, 96):
    z[0:30,i]=c*0.35/41 - 0.4
    c=c+1
    
# from 76 lower less
c=76
for i in range(76, 96):
    z[42:,i]=c*0.35/41 - 0.4
    c=c+1
    

z[0:30,91:96]=0.28
z[46:,91:96]=0.28


z[30:42,95:96]=0.7




def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)



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



def test_reward():
    state = env.reset()
    done = False
    total_reward = 0
    print('testing...')
    limit = 0
    while not done:
        state_input = K.expand_dims(state, 0)
        action_probs = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        action = np.random.choice(n_actions, p=action_probs[0, :])
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        limit += 1
        if limit > 128:
            break
    return total_reward


def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot



env = football_env.create_environment(env_name='tr1', representation='extracted', render=False)
env_help = football_env.create_environment(env_name='tr1', representation='simple115', render=False)


state = env.reset()
env_help.reset()

state_dims = env.observation_space.shape
n_actions = env.action_space.n

dummy_n = np.zeros((1, 1, n_actions))
dummy_1 = np.zeros((1, 1, 1))

tensor_board = TensorBoard(log_dir='./logs')


# Read the model to continue training or run predictions to see game
read_model=0
if read_model==1:
	

	oldpolicy_probs = np.zeros((1, n_actions))
	oldpolicy_probs=tf.cast(oldpolicy_probs, tf.float32)
	advantages = np.zeros((1, 1))
	advantages=tf.cast(advantages, tf.float32)
	rewards = np.zeros((1, 1))
	rewards=tf.cast(rewards, tf.float32)
	values = np.zeros((1, 1))
	values=tf.cast(values, tf.float32)	
	
	model_actor = tf.keras.models.load_model("model_actor_11_vs_11_easy_stochastic1000.hdf5",custom_objects={'loss': ppo_loss(oldpolicy_probs, advantages, rewards, values)})
	model_critic = load_model("model_critic_11_vs_11_easy_stochastic1000.hdf5", compile=True)
	
else:
	    model_actor = get_model_actor_extracted(input_dims=state_dims, output_dims=n_actions,n_actions=n_actions)
	    model_critic = get_model_critic_extracted(input_dims=state_dims)

ppo_steps = 1200 #220
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
    state_input = K.expand_dims(state, 0)
    start=1
    for itr in range(ppo_steps):
    
        action_dist = model_actor.predict([state_input, dummy_n, dummy_1, dummy_1, dummy_1], steps=1)
        #print("state_input) : {}".format(state_input.shape))
        q_value = model_critic.predict([state_input], steps=1)
        action = np.random.choice(n_actions, p=action_dist[0, :])
        action_onehot = np.zeros(n_actions)
        action_onehot[action] = 1

        observation, reward, done, info = env.step(action)
        observation_help, reward_help, done_help, info_help = env_help.step(action)
 
               
        mask = not done
        #print("state[87+2]) : {}".format(state[87+2]))  state_input) : (1, 72, 96, 3)


        states.append(state)
        actions.append(action)
        actions_onehot.append(action_onehot)
        values.append(q_value)
        masks.append(mask)
        actions_probs.append(action_dist)

        state = observation
        state_input=np.expand_dims(state, axis=0)
        state_input=state_input/255
        #print('state_input.max() {}'.format(state_input.max()))


        if reward_help==1 or reward_help==-1:
        	pass
        else:
        	if observation_help[87+7]!=0:
        		reward=0.1           


        #print("observation_help[87+7] : {}".format(observation_help[87+7]))
        if reward==1 or reward==-1:
        	pass
        elif start==0:
                m=np.sum(np.multiply(state_input[:,:,:,2][0],np.expand_dims(z, axis=0)[0]))
                reward = reward+m

        rewards.append(reward)


        print('itr: ' + str(itr) + ', action=' + str(action) + ', reward=' + str(reward) + ', q val=' + str(q_value))

        if done:
            env.reset()
            env_help.reset()
            start=1
        else:
            start=0

        if done_help:
            env.reset()
            env_help.reset()
            start=1
        else:
            start=0



    q_value = model_critic.predict(state_input, steps=1)
    values.append(q_value)
    returns, advantages = get_advantages(values, masks, rewards)
    actor_loss = model_actor.fit(
        [np.array(states), np.array(actions_probs), np.array(advantages), np.array(np.reshape(rewards, newshape=(-1, 1, 1))), np.array(values[:-1])],
        [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8)
    critic_loss = model_critic.fit(np.array(states), np.array(np.reshape(returns, newshape=(-1, 1))), shuffle=True, epochs=8,
                                  verbose=True)

    if iters % 20 == 0 and iters!=0:
	    avg_reward = np.mean([test_reward() for _ in range(5)])
	    print('total test reward=' + str(avg_reward))
	    if avg_reward > best_reward:
	        print('best reward=' + str(avg_reward))
	        model_actor.save('model_actor_11_vs_11_easy_stochastic_extracted__2{}_{}.hdf5'.format(iters, avg_reward))
	        model_critic.save('model_critic_11_vs_11_easy_stochastic_extracted_2{}_{}.hdf5'.format(iters, avg_reward))
	        best_reward = avg_reward
	    if best_reward > 0.9 or iters > max_iters:
	        target_reached = True
	    
	    
	    model_actor.save('model_actor_11_vs_11_easy_stochastic_extracted__2{}.hdf5'.format(iters),save_format='tf')
	    model_critic.save('model_critic_11_vs_11_easy_stochastic_extracted__2{}.hdf5'.format(iters))
    iters += 1
    env.reset()



env.close()
