from kaggle_environments import make,evaluate
from _11_submission import agent,states_converter
from _11conv_DQN_Nets import DQNAgent
import os


#Environment settings
env = make("connectx", debug=True)
trainer = env.train([None, "random"])
#Board size
configuration={"rows": 6, "columns": 7}
rows=configuration['rows']
columns=configuration['columns']


#Training process settings
EPISODES=30
#agent2=DQNAgent(rows=rows,columns=columns,action_size=columns)

verbose=False

obs = trainer.reset()
done=False
for e in range(EPISODES):
    state = trainer.reset()
    if verbose: print("episode: ",e)

    for i in range(100):    #episode isnt finished
        action = agent(state,configuration) # Action for the agent being trained.
        new_state, reward, done, info = trainer.step(action)
        #print('new_state=', new_state)
        state=new_state
        if done:
            #print(states_converter(new_state,rows,columns))
            print("Episode: {}, reward: {} after {} moves".format(e,reward,i))
            break

def mean_reward(rewards):
    rewards_corrected=[]
    for r in rewards:
        if r[0] is None: r[0]=0 #removing None Type result if game result is draw
        rewards_corrected.append(r[0])

    return sum(r for r in rewards_corrected) / float(len(rewards_corrected))

# Run multiple episodes to estimate its performance.
rewards=evaluate("connectx", [agent, 'random'], num_episodes=30)
print('rewards=',rewards)
print("My Agent vs Random Agent:", mean_reward(rewards))
os.remove('./model_action_predictor.h5')