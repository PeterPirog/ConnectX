from kaggle_environments import make
from _11_submission import agent,states_converter
from _11conv_DQN_Nets import DQNAgent


#Environment settings
env = make("connectx", debug=True)
trainer = env.train([None, "random"])
#Board size
configuration={"rows": 6, "columns": 7}
rows=configuration['rows']
columns=configuration['columns']


#Training process settings
EPISODES=20
agent2=DQNAgent(rows=rows,columns=columns,action_size=columns)

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