import random
import pickle
import gym
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

from kaggle_environments import evaluate, make, utils

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
        model.summary()
        return model

    def memorize(self,state,action,reward,next_state,done):
        #reward=-100*next_state[0,2]**2
        #print(reward)
        self.memory.append((state,action,reward,next_state,done))

    def act(self,state):
        if np.random.rand()<=self.epsilon:
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0]) #returns action

    def replay(self,batch_size):
        minibatch=random.sample(self.memory,batch_size)
        states = np.zeros((batch_size,42))  # 4 -> numbers in state vector
        actions=np.zeros((batch_size,1))
        next_states=np.zeros((batch_size,42))
        targets=np.zeros((batch_size,7)) #number of sctions

        idx=0
        for state,action,reward,next_state,done in minibatch:


            states[idx,:]=state
            actions[idx,:]=action
            next_states[idx,:]=next_state
            if not done:
                target=(reward+self.gamma*np.amax(self.model.predict(next_state)[0]))
            else:
                target = reward

            target_f=self.model.predict(state)
            #print('target=', target_f)
            target_f[0][action]=target

            targets[idx,:]=target_f
            #print('targets=', targets)
            idx+=1
        self.model.train_on_batch(x=states,y=target_f)        # fit(state,target_f,epochs=1,verbose=0)


        if self.epsilon>self.epsilon_min:
            self.epsilon*=self.epsilon_decay

    def load(self,name):
        self.model.load_weights(name)

    def save(self,name):
        self.model.save(filepath=name)
        #self.model.save_weights(name)

#Define function to convert state
def state2observation(state):

    board=state['board']
    board=np.array(board)
    mark=state['mark']
    if mark==1:
        board[board==1]=5
        board[board==2]=-5
    if mark==2:
        board[board==1]=-5
        board[board==2]=5
    board=board/5
    return board

if __name__=="__main__":


    env = make("connectx", debug=True)
    # Play as first position against random agent.
    env = env.train([None, "random"])

    #Definie init parameters
    state_size=42
    action_size=7
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
        #print('raw state=',state)
        state=state2observation(state)

        for time in range(200):

            action=agent.act(state)
            next_state,reward,done,__=env.step(action)

            #conversions
            next_state=state2observation(next_state)
            print('reward=', reward)



            agent.memorize(state,action,reward,next_state,done)
            state=next_state
            if done:
                print('episode: {}/{}, score: {}, e: {:.2}'.format(e,EPISODES,time,agent.epsilon))
                break
            if len(agent.memory)>batch_size:
                agent.replay(batch_size)
        if e%10==0:
            agent.save('whole_model.obj')
            #agent.save('cartpole-dqn.h5')
    sars=agent.memory
    print(sars)

    pickle.dump(sars, open('sars_500_episodes.obj', 'wb') )

