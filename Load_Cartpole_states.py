# File sars_1000_episodes.obj has sars history (state,action,reward,next_state,done)
import pickle
import numpy as np
import datetime
import os


from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import  Dense
from tensorflow.keras.optimizers import Adam
from collections import deque


def list1D_2_features(list1D):
    feature_vector = np.stack(list1D)
    feature_vector = feature_vector[np.newaxis].transpose()
    return feature_vector

def list2D_2_features(list2D):
   feature_array = np.stack(list2D, axis=1)
   feature_array = np.squeeze(feature_array, axis=0)
   return feature_array

def sarsObject_2_sarsFeatures(sars_object,verbose=False):
    # unziping whole list to separate lists
    list_sars = list(zip(*sars_object))

    list_state = list_sars[0]
    list_action = list_sars[1]
    list_reward = list_sars[2]
    list_next_state = list_sars[3]
    list_done = list_sars[4]

    states = list2D_2_features(list_state)  # STATES
    next_states = list2D_2_features(list_next_state) # NEXT STATES
    actions = list1D_2_features(list_action) # ACTIONS
    rewards = list1D_2_features(list_reward) # REWARDS
    done = list1D_2_features(list_done)  # REWARDS

    if verbose:
        print('Feature arrays size:')
        print('states.shape=', states.shape)
        print('next_states=', next_states.shape)
        print('actions=', actions.shape)
        print('rewards=', rewards.shape)
        print('done=', done.shape)
    return states, actions,rewards,next_states,done

def sarsFile_2_sarsFeatures(sarsFilePath,verbose=False):
    filehandler = open(sarsFilePath, 'rb')
    object = pickle.load(filehandler)
    states, actions, rewards, next_states, done = sarsObject_2_sarsFeatures(object,verbose=verbose)
    pickle.dump([states, actions, rewards, next_states, done], open('sars_features.obj', 'wb'))
    return states, actions, rewards, next_states, done


def splitData(dataX,dataY,train_ratio = 0.75,validation_ratio = 0.15,test_ratio = 0.10,random_state=42):

    # train is now 75% of the entire data set
    # the _junk suffix means that we drop that variable completely
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=1 - train_ratio,random_state = random_state)
    # test is now 10% of the initial data set
    # validation is now 15% of the initial data set
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test,test_size=test_ratio / (test_ratio + validation_ratio),random_state = random_state)

    print(x_train, x_val, x_test)
    return x_train, x_val, x_test, y_train, y_val, y_test
####################################################################################
class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        #self.memory=deque()
        self.gamma=0.95 #discount rate
        self.epsilon=1.0 #exploration rate
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        self.modelQ=self.__build_model_Q()
        self.modelEnc = self.__build_model_Enc()
        self.entropy=0


    def __build_model_Q(self):
        #Neural Net for Deep-Q learning Model
        inputs = tf.keras.Input(shape=(self.state_size,), name='input_states')
        x=layers.LayerNormalization(name='normalization_layer')(inputs)
        x = layers.Dense(24, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer1')(x)
        x = layers.Dense(24, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer2')(x)
        outputs = layers.Dense(self.action_size,activation='elu',kernel_initializer='glorot_normal',name='output_layer')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Q_values_model')
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model

    def __build_model_Enc(self):
        #Neural Net for Deep-Q learning Model
        inputs = tf.keras.Input(shape=(self.state_size,), name='input_states')
        x=layers.LayerNormalization(name='normalization_layer')(inputs)
        x = layers.Dense(24, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer1')(x)
        x = layers.Dense(24, activation='elu',kernel_initializer='glorot_normal',name='hidden_layer2')(x)
        outputs = layers.Dense(self.state_size,activation='elu',kernel_initializer='glorot_normal',name='output_layer')(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Encoder_model')
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()
        return model
#################################################################################################################
if __name__== "__main__":
    states, actions, rewards, next_states, done = sarsFile_2_sarsFeatures('sars_1000_episodes.obj',verbose=True)

    x_train, x_val, x_test, y_train, y_val, y_test=splitData(dataX=states,dataY=states)
    print(x_train.shape)

    agent=DQNAgent(state_size=4,action_size=2)

    # Clear any logs from previous runs
    #os.system('rm - rf. / logs /')

    log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir=r'C:\Users\Ila\PycharmProjects\DQN\logs\fit'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history=agent.modelEnc.fit(x=x_train,y=y_train,validation_data=(x_val,y_val),
                       batch_size=32,epochs=1,callbacks=[tensorboard_callback])

    #tensorboard --logdir logs/fit
#tensorboard --logdir=C:\Users\Ila\PycharmProjects\DQN\logs\fit


