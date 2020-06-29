from kaggle_environments import make, evaluate
from submission import agent, ConX
import gym
import torch as T
from DQN_01_base import Agent

import numpy as np




#############################################################################################################


if __name__ == '__main__':

    env = make("connectx", debug=True)
    env = env.train([None,'random'])#'random'
    """
    Cx = ConX(vector_form=True, convolution_ready=False)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=7,
                  eps_end=0.01, input_dims=Cx.input_size, lr=0.003)
    """
    scores, eps_history = [], []

    n_games = 50

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()

        while not done:
            action = agent(observation,configuration=None)
            observation_, reward, done, info = env.step(action)
            if reward is None: reward=0 # convert to 0 if draw
            score += reward
            observation=observation_
        scores.append(score)

        avg_score = np.mean([scores[-100:]])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)


