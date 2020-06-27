from random import choice
from tensorflow.keras import Model, layers

def agent(observation, configuration):
    rows=configuration['rows']
    columns=configuration['columns']
    inarow=configuration['inarow']
    #print(observation.board)
    action=choice([c for c in range(columns) if observation.board[c] == 0])
    return action

class AgentNet(Model):
    def __init__(self,):
        pass