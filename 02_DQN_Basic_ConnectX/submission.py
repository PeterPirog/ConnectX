import numpy as np
import torch as T
import requests
import os

"""
# original agent
def act(observation, configuration):
    board = observation.board
    columns = 7  # configuration.columns
    return [c for c in range(columns) if board[c] == 0][0]

def agent2(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]
"""

def agent(observation, configuration):
    rows = configuration.rows #6  # configuration['rows']
    columns = configuration.columns #7  # configuration['columns']

    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    if os.path.isfile('Cx_agent_net.pt')==False:
        r = requests.get('https://raw.githubusercontent.com/PeterPirog/tf_agents_tests/master/Cx_agent_net.pt')
        with open('Cx_agent_net.pt', 'wb') as f:
            f.write(r.content)
        while os.path.isfile('Cx_agent_net.pt')==False:
            pass


   # global model_Q
    model_Q=T.load('Cx_agent_net.pt',map_location=device)
    model_Q.eval()  #turn off droput, noise etc..


    Cx = ConX(rows=rows,columns=columns,vector_form=True, convolution_ready=False)
    state=Cx.obs2state(observation)

    state=T.tensor(state,dtype=T.float32,device=device)
    prediction=model_Q.forward(state)


    # Choose maximum Q-value action
    action = T.argmax(prediction)
    action = action.item()  # converting numpy int to native python int

    action = Cx.get_available_action(state, action)  # check if column is full
    return action


class ConX():
    def __init__(self, rows=6, columns=7, vector_form=True, convolution_ready=False, diagonal_features=False):
        self.rows = rows
        self.columns = columns
        self.n_actions = self.columns
        self.vector_form = vector_form  # True if 1D vector form observation else 2D array
        self.convolution = convolution_ready  # True i
        self.diagonal_features = diagonal_features

        # define neural network input size
        if self.vector_form: self.input_size = [self.rows * self.columns]
        if self.vector_form == False and self.convolution == False: self.input_size = [self.rows, self.columns]
        if self.vector_form == False and self.convolution == True: self.input_size = [self.rows, self.columns, 1]

    def obs2state(self, raw_observation):
        self.raw_observation = raw_observation  # include vector and mark
        self.obs_vect = np.array(self.raw_observation['board'], dtype=np.float32)
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

    def conv_obs_rwd_done_inf(self, observation, reward, done, info):
        observation = self.obs2state(observation)

        if reward is None:
            reward = 0.0
        else:
            reward = float(reward)

        return observation, reward, done, info

    def get_available_action(self, state, action):
        state=state.cpu().numpy()
        #print('state=', state)
        # reducink dimmensions
        vector = np.ndarray.flatten(state)
        short_vector = vector[:self.n_actions]
        idx=np.where(short_vector ==0)
        #print('short wector=', short_vector)
        #print('idx=',idx[0])
        if short_vector[action]==0:
            return action
        else:
            return np.random.choice(idx[0]).item()

        # print('short vector=',short_vector)
        idx = np.where(short_vector == 0)

        return action
