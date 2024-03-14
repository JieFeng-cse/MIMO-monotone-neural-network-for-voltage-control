# By Jie Feng, UCSD. This file is used to test the trained controllers.
import numpy as np
import torch
from tqdm import tqdm

from env_13bus import IEEE13bus, create_13bus
from safe_ddpg import ValueNetwork, DDPG
from distributed_controller import distributed_policy

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
seed = 13
torch.manual_seed(seed)

vlr = 1e-3
plr = 1e-3
ph_num = 1
max_ac = 0.2
epsilon = 0.02

pp_net = create_13bus()
injection_bus = np.array([2, 7, 8, 9, 12])
# injection_bus = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
env = IEEE13bus(pp_net, injection_bus)
num_agent = 3
num_controlled_bus = len(injection_bus)

obs_dim = env.obs_dim
action_dim = env.action_dim

hidden_dim = 200

type_name = 'single-phase'

policy_net = distributed_policy(env, obs_dim*5, action_dim*5, hidden_dim).to(device)
target_policy_net = distributed_policy(env, obs_dim*5, action_dim*5, hidden_dim).to(device)
value_net  = ValueNetwork(obs_dim=obs_dim*5, action_dim=action_dim*5, hidden_dim=hidden_dim).to(device)
target_value_net  = ValueNetwork(obs_dim=obs_dim*5, action_dim=action_dim*5, hidden_dim=hidden_dim).to(device)

for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)

agent = DDPG(policy_net=policy_net, value_net=value_net,
                target_policy_net=target_policy_net, target_value_net=target_value_net, value_lr=vlr, policy_lr=plr)
policynet_dict = torch.load(f'checkpoints/{type_name}/13bus/centralized/policy_net_checkpoint.pth')
agent.policy_net.load_state_dict(policynet_dict)

############################# TEST PERFORMANCE ##############################################
def test_performance():
    num_of_traj = 100
    num_steps = 100

    print(f'************Test Result*************************')
    rewards = []
    action_rewards = []
    voltage_rewards = []

    final_rewards = []
    final_action_rewards = []
    final_voltage_rewards = []
    for episode in tqdm(range(num_of_traj)):

        state = env.reset(seed = episode)
        episode_reward = 0
        episodic_action_reward = 0
        episodic_voltage_reward = 0
        last_action = np.zeros((5,)) #if single phase, 1, else ,3

        for step in range(num_steps):
            action = []
            action_tmp = agent.policy_net.get_action(np.asarray(state))
            action = last_action + epsilon*(action_tmp-last_action)

            # execute action a_t and observe reward r_t and observe next state s_{t+1}
            next_state, reward, reward_action,reward_voltage, done = env.step_eval(action)
            
            state = np.copy(next_state)
            episode_reward += reward  
            episodic_action_reward += reward_action
            episodic_voltage_reward += reward_voltage
            last_action = np.copy(action)

        rewards.append(episode_reward)
        action_rewards.append(episodic_action_reward)
        voltage_rewards.append(episodic_voltage_reward)

        final_rewards.append(reward)
        final_action_rewards.append(reward_action)
        final_voltage_rewards.append(reward_voltage)

    print('Episodic Rewards: ',np.mean(rewards),np.std(rewards))
    print('Episodic Action Rewards: ',np.mean(action_rewards),np.std(action_rewards))
    print('Episodic Voltage Rewards: ',np.mean(voltage_rewards),np.std(voltage_rewards))

    print('Steady-State Rewards: ',np.mean(final_rewards),np.std(final_rewards))
    print('Steady-State Action Rewards: ',np.mean(final_action_rewards),np.std(final_action_rewards))
    print('Steady-State Voltage Rewards: ',np.mean(final_voltage_rewards),np.std(final_voltage_rewards))


def test_monotonicity():
    state = env.reset(seed = 0)
    for i in range(1000):
        x1 = np.random.uniform(0.8,1.2,state.shape)
        x2 = np.random.uniform(0.8,1.2,state.shape)
        
        action_1 = agent.policy_net.get_action(x1)
        action_2 = agent.policy_net.get_action(x2)
        

        tmp = np.dot((action_1-action_2), (x1-x2))
        if tmp>0:
            print('violation!!!!!!!!!!!')

test_monotonicity()
test_performance()
