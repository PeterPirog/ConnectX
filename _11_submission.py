from random import choice
import numpy as np
from tensorflow.keras import Model, layers

def agent(observation, configuration):
    rows=configuration['rows']
    columns=configuration['columns']
    state=states_converter(observation,rows=rows,columns=columns)

    print('\n',state)
    action=choice([c for c in range(columns) if observation.board[c] == 0])
    return action


def states_converter(observation,rows=6,columns=7):
    board=observation['board']
    board=np.array(board)
    mark=observation['mark']
    if mark==1:
        board[board==1]=5
        board[board==2]=-5
    if mark==2:
        board[board==1]=-5
        board[board==2]=5
    board=board/5
    state=np.expand_dims(board,axis=0)
    state=state.reshape((rows,columns))
    return state


