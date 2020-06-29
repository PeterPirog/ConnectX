import pickle


def load_memory(filename='memory.pkl'):
    with open(filename, 'rb') as f:
        state_memory, action_memory, reward_memory, new_state_memory, terminal_memory = pickle.load(f)
    return state_memory, action_memory, reward_memory, new_state_memory, terminal_memory


state_memory, action_memory, reward_memory, new_state_memory, terminal_memory = load_memory()
# print(state_memory, action_memory, reward_memory, new_state_memory, terminal_memory)
print('state shape', state_memory[500])

with open('agent_net.pkl', 'rb') as f:
    net= pickle.load(f)



print(net)

import torch as T
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

a = [-0.12977114, 0.83363044, -0.17948864, -0.69042355, 0.15261325, -0.01526884, 0., 0.]
A=T.tensor(a,device=device)
print(net(A))