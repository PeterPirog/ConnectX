from random import choice
from tensorflow.keras import Model, layers

def agent(observation, configuration):
    rows=configuration['rows']
    columns=configuration['columns']
    inarow=configuration['inarow']
    #print(observation.board)
    return choice([c for c in range(columns) if observation.board[c] == 0])

class AgentNet(Model):
    def __init__(self,):
        pass