import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.optim as optim

# DPPG class
class DDPG:
    def __init__(self, policy_net, value_net,
                 target_policy_net, target_value_net,
                 value_lr=2e-4,
                 policy_lr=1e-4, epsilon=0.01):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.policy_net = policy_net
        self.value_net = value_net
        self.target_policy_net = target_policy_net
        self.target_value_net = target_value_net
        self.epsilon = epsilon
        
        self.value_lr = value_lr
        self.policy_lr = policy_lr
        
        self.value_optimizer = optim.Adam(value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)
        self.value_criterion = nn.MSELoss()

    def train_step(self, replay_buffer, batch_size, 
                   gamma=0.99,
                   soft_tau=1e-2):
        # sample state pairs
        state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        # update the value network
        next_action = action+self.epsilon*(self.target_policy_net(next_state)-action)    
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + gamma*(1.0-done)*target_value
        
        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
 
        self.value_optimizer.step()
        # update the policy network (Monotone network)
        policy_loss = self.value_net(state, self.policy_net(state)) 
        policy_loss =  -policy_loss.mean() 
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update the policy network
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data*soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )


# standard ddpg policy network (not deployed)
class PolicyNetwork(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.linear1 = nn.Linear(obs_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        state.requires_grad = True
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]


# value network
class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(obs_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class ReplayBufferPI:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, last_action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, last_action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, last_action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

