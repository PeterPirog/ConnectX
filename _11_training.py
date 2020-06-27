from kaggle_environments import make
from _11_submission import agent,states_converter
from _11_DQN_Nets import DQNAgent


#Environment settings
env = make("connectx", debug=True)
configuration={"rows": 6, "columns": 7}
trainer = env.train([None, "random"])

#Training process settings
EPISODES=10
agent2=DQNAgent(rows=configuration['rows'],columns=configuration['columns'],action_size=configuration['columns'])

verbose=True

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
            print(states_converter(new_state))
            print('reward=', reward)
            break