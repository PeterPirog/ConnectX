import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle


class DeepNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.lr = lr
        # Define layers
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        #self.flatt=nn
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01,
                 eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.iteration_memory_save = 10000  # save collected values  and save agent after number of iterations
        self.filename_memory = 'Cx_memory.pkl'  # filename to store collected states
        self.filename_agent_net = 'Cx_agent_net.pt'  # filename to save trained agent

        #self.Q_eval = self.load_agent_net()
        try:  # continue learning if possible
            self.Q_eval = self.load_agent_net()
            print('PyTorch net checkpoint loaded')
        except:
            self.Q_eval = DeepNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=500, fc2_dims=256)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

        if self.mem_cntr % self.iteration_memory_save == 0:
            print('Collected {} states. Agent net and collected data saved'.format(str(self.mem_cntr)))
            self.save_memory()
            self.save_agent_net()

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return int(action)

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]  # action batch beacuse we get only values for action taken
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

    def save_memory(self):

        with open(self.filename_memory, 'wb') as f:
            pickle.dump([self.state_memory, self.action_memory, self.reward_memory, self.new_state_memory,
                         self.terminal_memory], f)

    def load_memory(self, filename='memory.pkl'):
        self.filename_memory = filename
        with open(filename, 'rb') as f:
            state_memory, action_memory, reward_memory, new_state_memory, terminal_memory = pickle.load(f)
        return state_memory, action_memory, reward_memory, new_state_memory, terminal_memory

    def save_agent_net(self):
        T.save(self.Q_eval,self.filename_agent_net)

        #with open(self.filename_agent_net, 'wb') as f:
         #   pickle.dump(self.Q_eval, f)

    def load_agent_net(self):
        q_network=T.load(self.filename_agent_net)
        #with open(self.filename_agent_net, 'rb') as f:
            #q_network = pickle.load(f)
        return q_network
