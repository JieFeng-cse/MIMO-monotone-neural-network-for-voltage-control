# By Jie Feng, UCSD. This file creates the simulation environment for IEEE 13-bus system with pandapower
import numpy as np
from numpy import linalg as LA
import gym
import matplotlib.pyplot as plt

from scipy.io import loadmat
import pandapower as pp

'''
This class define a IEEE 13-bus system simulation, the model is based on the matpower configuaration available in folder pandapower models.
Overview: we have four functions __init__(); step(); step_eval(); and reset().
step() and step_eval() use action as input, and gives you the next state. reset() can initialize the system with disturbance.

'''
class IEEE13bus(gym.Env):
    def __init__(self, pp_net, injection_bus, v0=1, vmax=1.05, vmin=0.95, all_bus=False):
        self.network =  pp_net
        self.obs_dim = 1
        self.action_dim = 1
        self.injection_bus = injection_bus
        self.agentnum = len(injection_bus)
        self.v0 = v0 
        self.vmax = vmax-0.01
        self.vmin = vmin+0.01
        
        # Initialize the real/reactive power profile
        self.load0_p = np.copy(self.network.load['p_mw'])
        self.load0_q = np.copy(self.network.load['q_mvar'])

        self.gen0_p = np.copy(self.network.sgen['p_mw'])
        self.gen0_q = np.copy(self.network.sgen['q_mvar'])
        
        self.state = np.ones(self.agentnum, )
        self.all_bus = all_bus

    
    def step(self, action): 
        # you can custimize the reward if needed, low voltage might rise a more damaging issue, so in training we give a higher cost.
        done = False 
        # global reward
        reward = float(-10*LA.norm(action,2) -200*LA.norm(np.clip(self.state-self.vmax, 0, np.inf))
                       - 240*LA.norm(np.clip(self.vmin-self.state, 0, np.inf)))
        # extra cost if deviation is too large
        if np.max(self.state)>1.2:
            reward -= float(500*LA.norm(np.clip(self.state-self.vmax, 0, np.inf)))
        elif np.min(self.state)<0.8:
            reward -= float(500*LA.norm(np.clip(self.vmin-self.state, 0, np.inf)))
        agent_num = len(self.injection_bus)
        reward_sep = np.zeros(agent_num, )

        # apply contol action to the controlled buses
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 
        # run power flow simulation
        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        # get the voltage after the control action
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        # if voltage is restored, well done!
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        return self.state, reward, reward_sep, done
    
    def step_eval(self, action): 
        # you can custimize the reward if needed, this part is used for evaluation.
        done = False 
        
        reward = float(-10*LA.norm(action,2) -200*LA.norm(np.clip(self.state-self.vmax-0.01, 0, np.inf))
                       - 200*LA.norm(np.clip(self.vmin-0.01-self.state, 0, np.inf)))
        # seperate reward
        reward_action = float(-10*LA.norm(action,2))
        reward_voltage = float( -200*LA.norm(np.clip(self.state-self.vmax-0.01, 0, np.inf))
                       - 200*LA.norm(np.clip(self.vmin-0.01-self.state, 0, np.inf)))
    
        for i in range(len(self.injection_bus)):
            self.network.sgen.at[i, 'q_mvar'] = action[i] 

        pp.runpp(self.network, algorithm='bfsw', init = 'dc')
        
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        
        if(np.min(self.state) > 0.95 and np.max(self.state)< 1.05):
            done = True
        return self.state, reward, reward_action,reward_voltage, done
    
    def reset(self, seed=1): #sample different initial volateg conditions during training
        np.random.seed(seed)
        senario = np.random.choice([0,1])
        
        if(senario == 0):#low voltage 
           # Low voltage
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            # introduce random noise to generation profiles to create voltage deviation between -15% to -5%
            self.network.sgen.at[0, 'p_mw'] = -0.5*np.random.uniform(1, 7)
            self.network.sgen.at[1, 'p_mw'] = -0.8*np.random.uniform(1, 4)
            self.network.sgen.at[2, 'p_mw'] = -0.3*np.random.uniform(1, 4)
            self.network.sgen.at[3, 'p_mw'] = -0.3*np.random.uniform(1, 2)
            self.network.sgen.at[4, 'p_mw'] = -0.3*np.random.uniform(1, 2)
            if self.all_bus:
                for i in range(len(self.injection_bus)):
                    self.network.sgen.at[i, 'p_mw'] = -0.3*np.random.uniform(1, 2.5)
        elif(senario == 1): #high voltage 
            self.network.sgen['p_mw'] = 0.0
            self.network.sgen['q_mvar'] = 0.0
            self.network.load['p_mw'] = 0.0
            self.network.load['q_mvar'] = 0.0
            # introduce random noise to generation profiles to create voltage deviation between 5% to 15%
            self.network.sgen.at[0, 'p_mw'] = np.random.uniform(0.5, 2)
            self.network.sgen.at[1, 'p_mw'] = np.random.uniform(0, 2.51)
            self.network.sgen.at[2, 'p_mw'] = np.random.uniform(0, 2)

            self.network.sgen.at[3, 'p_mw'] = 0.3*np.random.uniform(0, 0.2)
            self.network.sgen.at[4, 'p_mw'] = 0.5*np.random.uniform(2, 3)
            self.network.sgen.at[5, 'q_mvar'] = 0.4*np.random.uniform(0, 10)
            
            self.network.sgen.at[10, 'p_mw'] = np.random.uniform(0.2, 3)
            self.network.sgen.at[11, 'p_mw'] = np.random.uniform(0, 1.5)
            #for all buses scheme
            if self.all_bus:
                self.network.sgen.at[6, 'p_mw'] = 0.5*np.random.uniform(1, 2)
                self.network.sgen.at[7, 'p_mw'] = 0.2*np.random.uniform(1, 3)
                self.network.sgen.at[8, 'p_mw'] = 0.2*np.random.uniform(2, 3)
                self.network.sgen.at[9, 'p_mw'] = np.random.uniform(0.1, 0.5)
            
        
        pp.runpp(self.network, algorithm='bfsw')
        self.state = self.network.res_bus.iloc[self.injection_bus].vm_pu.to_numpy()
        return self.state

def create_13bus():
    # create the pandapower simulation instance and create generators at controlled buses.
    pp_net = pp.converter.from_mpc('pandapower models/pandapower models/case_13.mat', casename_mpc_file='case_mpc')
    
    pp_net.sgen['p_mw'] = 0.0
    pp_net.sgen['q_mvar'] = 0.0

    pp.create_sgen(pp_net, 2, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 8, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 9, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 12, p_mw = 0, q_mvar=0)

    pp.create_sgen(pp_net, 1, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 4, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 5, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 6, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 10, p_mw = 0, q_mvar=0)
    pp.create_sgen(pp_net, 11, p_mw = 0, q_mvar=0)
    
    
    # In the original IEEE 13 bus system, there is no load in bus 3, 7, 8. 
    # Add the load to corresponding node for dimension alignment in RL training
    pp.create_load(pp_net, 3, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 7, p_mw = 0, q_mvar=0)
    pp.create_load(pp_net, 8, p_mw = 0, q_mvar=0)

    return pp_net

if __name__ == "__main__":
    # use to check the distribution of original voltage devaition created by reset.
    net = create_13bus()
    injection_bus = np.array([2, 7, 8, 9, 12])
    env = IEEE13bus(net, injection_bus)
    state_list = []
    for i in range(200):
        state = env.reset(i)
        state_list.append(state)
    state_list = np.array(state_list)
    fig, axs = plt.subplots(1, len(injection_bus), figsize=(15,3))
    for i in range(len(injection_bus)):
        axs[i].hist(state_list[:,i])
    plt.show()
    



