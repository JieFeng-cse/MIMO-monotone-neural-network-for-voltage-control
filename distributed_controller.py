# By Jie Feng, UCSD. Pytorch implementation of MIMO monotone network (as gradient of ICNN)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import linalg as LA
from icnn import ReHU, ICNN

'''
Base distributed policy, representing each g function
'''
class distributed_policy(nn.Module):
    def __init__(self, env, obs_dim, action_dim, hidden_dim,distribued=False):
        super(distributed_policy, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device   = torch.device("cuda" if use_cuda else "cpu")

        self.env = env
        self.rehu = ReHU(float(7.0))
        self.controller = ICNN([obs_dim, hidden_dim, hidden_dim, 1], activation=torch.nn.Softplus()) 
        self.distributed = distribued

    def forward(self, state):
        output = self.controller(state)
        # calculate the jacobian of ICNN
        compute_batch_jacobian = torch.vmap(torch.func.jacrev(self.controller))
        # get the action, which is monotonically decreasing.
        action = -compute_batch_jacobian(state).squeeze()*0.1 
        return action

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()