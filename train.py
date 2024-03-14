# By Jie Feng, UCSD. This file trains the MIMO monotone neural network for voltage control
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

from env_13bus import IEEE13bus, create_13bus
from safe_ddpg import ValueNetwork, DDPG, ReplayBufferPI
from distributed_controller import distributed_policy

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

seed = 13
torch.manual_seed(seed)


"""
Initialize the constants
"""
vlr = 2e-4 # learning rate
plr = 2e-4
ph_num = 1 # single phase dynamics
max_ac = 0.3 # for noise
epsilon = 0.02
# Note: control action is defined as  reactive power q(t+1)=q(t)+epsilon(action-q(t)), action is the output of the monotone network

pp_net = create_13bus()
injection_bus = np.array([2, 7, 8, 9, 12])
env = IEEE13bus(pp_net, injection_bus)
num_agent = len(injection_bus)

#action and state dim for each bus
obs_dim = env.obs_dim
action_dim = env.action_dim
hidden_dim = 200
type_name = 'single-phase'


# initialize the networks, 5 buses are controlled
policy_net = distributed_policy(env, obs_dim*num_agent, action_dim*num_agent, hidden_dim).to(device)
target_policy_net = distributed_policy(env, obs_dim*num_agent, action_dim*num_agent, hidden_dim).to(device)
value_net  = ValueNetwork(obs_dim=obs_dim*num_agent, action_dim=action_dim*num_agent, hidden_dim=hidden_dim).to(device)
target_value_net  = ValueNetwork(obs_dim=obs_dim*num_agent, action_dim=action_dim*num_agent, hidden_dim=hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

agent = DDPG(policy_net=policy_net, value_net=value_net,
                target_policy_net=target_policy_net, target_value_net=target_value_net, value_lr=vlr, policy_lr=plr, epsilon=epsilon)

replay_buffer = ReplayBufferPI(capacity=1000000)






# training episode
num_episodes = 2000    
# trajetory length each episode
num_steps = 50  
batch_size = 256

# To ease the training, we initialize with a linear controller which is u=-2(v-1) by supervised learning. (mimicing the linear controller)
policynet_dict = torch.load(f'checkpoints/{type_name}/13bus/centralized/policy_net_checkpoint_linear.pth')
agent.policy_net.load_state_dict(policynet_dict)

for target_param, param in zip(agent.target_policy_net.parameters(), agent.policy_net.parameters()):
    target_param.data.copy_(param.data)

rewards = []
avg_reward_list = []
for episode in range(num_episodes):
    # initialize the system
    state = env.reset(seed = episode)
    episode_reward = 0
    last_action = np.zeros((5,)) #if single phase, 1, else ,3

    for step in range(num_steps):
        action = []
        action_tmp = agent.policy_net.get_action(np.asarray(state))  + np.random.normal(0, max_ac)/np.sqrt(episode/5+1) #action noise
        action = last_action + epsilon*(action_tmp-last_action)

        # execute action a_t and observe reward r_t and observe next state s_{t+1}
        next_state, reward, reward_sep, done = env.step(action)
        
        if(np.min(next_state)<0.75): #if voltage violation > 25%, skip updation of models to avoid instability.
            continue
        else:
            state_buffer = state
            action_buffer = action
            last_action_buffer = last_action
            next_state_buffer = next_state
            # store transition (s_t, a_t, r_t, s_{t+1}) in R
            replay_buffer.push(state_buffer, action_buffer, last_action_buffer,
                                        reward, next_state_buffer, done) 
            
            if len(replay_buffer) > batch_size:
                agent.train_step(replay_buffer=replay_buffer, 
                                        batch_size=batch_size)

            if(done):
                episode_reward += reward  
            else:
                state = np.copy(next_state)
                episode_reward += reward    

        last_action = np.copy(action)

    rewards.append(episode_reward)
    avg_reward = np.mean(rewards[-40:])
    if(episode%200==0):
        print("Episode * {} * Avg Reward is ==> {}".format(episode, avg_reward))
    avg_reward_list.append(avg_reward)
pth_value = f'checkpoints/{type_name}/13bus/centralized/value_net_checkpoint.pth'
pth_policy = f'checkpoints/{type_name}/13bus/centralized/policy_net_checkpoint.pth'
# torch.save(agent.value_net.state_dict(), pth_value)
# torch.save(agent.policy_net.state_dict(), pth_policy)
