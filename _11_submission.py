from random import choice
import numpy as np
from tensorflow.keras import models
from tensorflow.python.lib.io import file_io

MODEL_Q = 0


def agent(observation, configuration):
    rows = configuration['rows']
    columns = configuration['columns']

    global MODEL_Q
    # print('MODEL_Q=', MODEL_Q)
    # converting state to 2D array
    state = states_converter(observation, rows=rows, columns=columns, convolution=True)
    if not state.shape == (1, rows, columns, 1):
        print("State array shape error, state shape =", state.shape)

    # Load agent model from file
    if MODEL_Q == 0:
        # model_file=file_io.FileIO('gs://bert-pl/ConnectX/model_action_predictor.h5',mode='rb')
        MODEL_Q = models.load_model('model_action_predictor.h5')

    # predict Q-values for current state

    prediction = MODEL_Q.predict(state)[0]
    # print('\npredictions=',prediction)

    # Choose maximum Q-value action
    action = np.argmax(prediction)
    action = np.int16(action).item()  # converting numpy int to native python int
    return action


# This function converts obserwation from form:
# {'board': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 2, 1, 0, 0, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 2], 'mark': 1}
# to 2D numpy array where 1- own marks, -1 oponent marks
def states_converter(observation, rows, columns, convolution=False):
    board = observation['board']
    board = np.array(board)
    mark = observation['mark']
    if mark == 1:
        board[board == 1] = 5
        board[board == 2] = -5
    if mark == 2:
        board[board == 1] = -5
        board[board == 2] = 5
    board = board / 5
    state = np.expand_dims(board, axis=0)
    state = state.reshape((1, rows, columns))
    if convolution:
        state = np.expand_dims(state, axis=3)
        # print('state shape=',state.shape)
    return state
