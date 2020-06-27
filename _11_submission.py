import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

MODEL_Q = 0  # global variable to checking if model is defined


def agent(observation, configuration):
    rows = configuration['rows']
    columns = configuration['columns']
    global MODEL_Q

    # converting state to 2D array
    state = states_converter(observation, rows=rows, columns=columns, convolution=True)
    if not state.shape == (1, rows, columns, 1):
        print("State array shape error, state shape =", state.shape)

    if not os.path.exists('./model_action_predictor.h5'):
        os.system("gsutil cp gs://bert-pl/ConnectX/model_action_predictor.h5 ./")
        while not os.path.exists('./model_action_predictor.h5'):
            pass

    if not isinstance(MODEL_Q, tf.keras.Model): #if model variable not exist
        MODEL_Q = models.load_model('./model_action_predictor.h5')

    # JSON TEST
    # json_model=MODEL_Q.to_json()

    # predict Q-values for current state
    prediction = MODEL_Q.predict(state)[0]

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
    return state
