from kaggle_environments import make, evaluate
import gym
import torch as T
from DQN_01_base import Agent
from submission_strNet import DeepNetwork #this network is for cpu

import numpy as np


class ConX():
    def __init__(self, rows=6, columns=7, vector_form=True, convolution_ready=False,diagonal_features=False):
        self.rows = rows
        self.columns = columns
        self.n_actions = self.columns
        self.vector_form = vector_form  #True if 1D vector form observation else 2D array
        self.convolution = convolution_ready #True i
        self.diagonal_features=diagonal_features

        # define neural network input size
        if self.vector_form: self.input_size = [self.rows * self.columns]
        if self.vector_form == False and self.convolution == False: self.input_size = [self.rows, self.columns]
        if self.vector_form == False and self.convolution == True: self.input_size = [self.rows, self.columns, 1]

    def obs2state(self, raw_observation):
        self.raw_observation = raw_observation  # include vector and mark
        self.obs_vect = np.array(self.raw_observation['board'],dtype=np.float32)
        self.obs_board = None
        self.obs_conv = None
        self.mark = self.raw_observation['mark']

        if self.mark == 1:
            self.obs_vect[self.obs_vect == 1] = 5
            self.obs_vect[self.obs_vect == 2] = -5
        if self.mark == 2:
            self.obs_vect[self.obs_vect == 1] = -5
            self.obs_vect[self.obs_vect == 2] = 5
        self.obs_vect = self.obs_vect / 5

        if self.vector_form:  # if single vector state form
            return self.obs_vect
        else:  # if 2D array form
            self.obs_board = np.expand_dims(self.obs_vect, axis=0)
            self.obs_board = self.obs_board.reshape((1, self.rows, self.columns))
            if not self.convolution:  # check if need preparing  for convolution , channels should be  added
                return self.obs_board
            else:
                self.obs_conv = np.expand_dims(self.obs_board, axis=3)
                return self.obs_conv
    def conv_obs_rwd_done_inf(self,observation, reward, done, info ):
        observation=self.obs2state(observation)

        if reward is None:
            reward=0.0
        else:
            reward=float(reward)

        return observation, reward, done, info

    def get_available_action(self,state,action):
        # reducink dimmensions
        vector=np.ndarray.flatten(state)
        short_vector=vector[:self.n_actions]
        #print('short vector=',short_vector)
        idx =np.where(short_vector ==0)
        if action in short_vector:
            return action
        else:
            action = int(np.random.choice(idx[0]))
            return action

#############################################################################################################


if __name__ == '__main__':

    env = make("connectx", debug=True)
    env = env.train([None, 'random'])

    Cx = ConX(vector_form=True, convolution_ready=False)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=7,
                  eps_end=0.01, input_dims=Cx.input_size, lr=0.003)
    scores, eps_history = [], []

    n_games = 200000

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        observation = Cx.obs2state(observation)

        while not done:
            action = agent.choose_action(observation)
            action=Cx.get_available_action(observation,action) #check if collumn is full

            observation_, reward, done, info = env.step(action)
            observation_, reward, done, info=Cx.conv_obs_rwd_done_inf(observation_, reward, done, info) #state conversions

            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean([scores[-100:]])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        #SAVING  Nets
        if i % 20 == 0:
            T.save(agent, 'Cx_whole_agent.pt')
            T.save(agent.Q_eval.state_dict(), 'Cx_agent_net_dict.pth') #Thic model is for GPU

            cpu_model = DeepNetwork(lr=0.003, n_actions=7, input_dims=[42], fc1_dims=500, fc2_dims=256) #This model is for CPU
            cpu_model.load_state_dict(T.load('Cx_agent_net_dict.pth', map_location=T.device('cpu')))
            #print('\n', cpu_model.state_dict())
            T.save(cpu_model.state_dict(), 'Cx_agent_net_dict_CPU.pth')

