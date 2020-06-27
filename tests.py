#https://www.kaggle.com/ajeffries/connectx-getting-started

from kaggle_environments import evaluate, make, utils
from submission import agent



env = make("connectx", debug=True)


# Play as first position against random agent.
trainer = env.train([None, "random"])

observation = trainer.reset()
print('env.configuration=',env.configuration)

while not env.done:
    my_action = agent(observation, env.configuration)
    #print("My Action", my_action)
    observation, reward, done, info = trainer.step(my_action)
    print("My Action", my_action)



def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [agent, agent], num_episodes=100)))
#print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [agent, "negamax"], num_episodes=10)))

print(evaluate("connectx", [agent, agent], num_episodes=100))

