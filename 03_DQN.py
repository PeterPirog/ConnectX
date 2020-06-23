import random
import pickle
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

#This script version save all sars in the file

EPISODES=500

class DQNAgent:
    def __init__(self,state_size,action_size):
        self.state_size=state_size
        self.action_size=action_size
        self.memory=deque()
        self.gamma=0.95 #discount rate
        self.epsilon=1.0 #exploration rate
        self.epsilon_min=0.01
        self.epsilon_decay=0.995
        self.learning_rate=0.001
        self.model=self.__build_model()

    def __build_model(self):
        #Neural Net for Deep-Q learning Model
        model=Sequential()
        model.add(Dense(24,input_dim=self.state_size,activation='relu'))
        model.add(Dense(24,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self,state,action,reward,next_state,done):
        reward=-100*next_state[0,2]**2
        #print(reward)
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0]) #returns action

    def replay(self,batch_size):
        minibatch=random.sample(self.memory,batch_size)
        for state,action,reward,next_state,done in minibatch:
            target=reward
            if not done:
                target=(reward+self.gamma*np.amax(self.model.predict(next_state)[0]))
            target_f=self.model.predict(state)
            target_f[0][action]=target
            self.model.fit(state,target_f,epochs=1,verbose=0)
        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save_weights(name)

if __name__=="__main__":
    env=gym.make('CartPole-v0')
    #Definie init parameters
    state_size=env.observation_space.shape[0]
    action_size=env.action_space.n
    agent=DQNAgent(state_size,action_size)
    #Yrying to load previous trained net if exist
    try:
        agent.load('cartpole-dqn.h5')
        print("agent file loaded")
    except:
        pass
    done=False
    batch_size=32

    for e in range(EPISODES):
        state=env.reset()
        state=np.reshape(state,[1,state_size])
        for time in range(1000):

            action=agent.act(state)
            next_state,reward,done,__=env.step(action)
            reward=reward if not done else -10
            next_state=np.reshape(next_state,[1,state_size])
            agent.memorize(state,action,reward,next_state,done)
            state=next_state
            if done:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e,EPISODES,time,agent.epsilon))
                break
            if len(agent.memory)>batch_size:
                agent.replay(batch_size)
        if e%10==0:
            agent.save('cartpole-dqn.h5')
    sars=agent.memory
    print(sars)

    pickle.dump(sars, open('sars_500_episodes.obj', 'wb') )

