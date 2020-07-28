import os
import gym
import numpy as np
import torch
import torch.nn as nn

# set params
learning_rate = 1e-6

class Model(nn.Module):
    def __init__(self, act_dim):
        hidden_size = act_dim*10

        self.imput_layer = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=act_dim)

    def forward(self, state):
        out = nn.functional.tanh(self.imput_layer(state))
        out = nn.functional.softmax(self.output_layer(state))
        return out

class Agent:
    def __init__(self, act_dim):
        self.act_dim = act_dim