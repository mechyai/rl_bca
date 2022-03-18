"""
Class for defining and managing all components relative to Branching Dueling Q-Network - BDQN
This BDQN will consist of:
- Double Q-learning
- Dueling Q-network
- Branching Q-network
- Experience Replay
"""

import os
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import time
import math
from collections import namedtuple, deque
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

""" 
-- Vanilla BDQN --
DQN:
Guides-
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://deeplizard.com/learn/video/PyQNfsGUnQA

Dueling DQN:
Guides-
https://www.youtube.com/watch?v=Odmeb3gkN0M&t=3s - Eden Meyer
Repos-
"""

# subclass tuple for experiences
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory(object):
    """Manages a replay memory, where data is stored as numpy arrays in a named tuple."""
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)  # faster than list

    def push(self, *args):
        """Save a transition to replay memory"""
        self.memory.append(Experience(*args))  # automatically pops if capacity is reached

    def sample(self):
        """Sample a transition randomly"""
        return random.sample(self.memory, self.batch_size)

    def can_provide_sample(self):
        """Check if replay memory has enough experience tuples to sample batch from"""
        return len(self.memory) >= self.batch_size


class BranchingQNetwork(nn.Module):
    """BDQ network architecture."""

    def __init__(self, observation_space, action_space, action_bins, shared_hidden_dim1, shared_hidden_dim2,
                 value_hidden_dim, action_hidden_dim):
        """
        Below, we define the BDQN architecture's network segments, consisting of a MLP shared representation,
        then a dueling state value module and individual advantage branches. At the end, the value and advantage streams
        are combined to get the branched Q-value output.
        """

        super().__init__()

        # Shared Representation (state feature estimator)

        self.shared_model = nn.Sequential(
            nn.Linear(observation_space, shared_hidden_dim1),
            # nn.ReLU(),
            nn.Linear(shared_hidden_dim1, shared_hidden_dim2),
            # nn.ReLU(),
        )

        # Value Stream
        self.value_head = nn.Sequential(
            nn.Linear(shared_hidden_dim2, value_hidden_dim),
            nn.Linear(value_hidden_dim, 1)
        )
        # Advantage Streams
        self.adv_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(shared_hidden_dim2, action_hidden_dim),
            nn.Linear(action_hidden_dim, action_bins)) for i in range(action_space)])

    def forward(self, x):
        # input
        out = self.shared_model(x)
        # out = self.connection(out)

        # branch
        value = self.value_head(out)
        advs = torch.stack([advantage_stream(out) for advantage_stream in self.adv_heads], dim=1)
        # recombine branches
        q_vals = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)  # identifiable method eqn #1

        return q_vals

    @staticmethod
    def _rescale_branched_gradient_hook(module, grad_input, grad_output):
        """Used to rescale the gradient converging at the action branches and value stream, into the shared network."""
        print(grad_output)
        print(torch.div(*grad_output, 1 + 4))
        return torch.div(*grad_output, 1 + 4)


class BranchingDQN(nn.Module):
    """A branching, dueling, & double DQN algorithm."""

    def __init__(self, observation_space, action_space, action_bins, target_update_freq, learning_rate, gamma,
                 shared_hidden_dim1, shared_hidden_dim2, state_hidden_dim, action_hidden_dim, td_target,
                 gradient_clip_norm, rescale_shared_grad_factor: float = None):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_bins = action_bins
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.rescale_shared_grad_factor = rescale_shared_grad_factor

        self.policy_network = BranchingQNetwork(observation_space, action_space, action_bins, shared_hidden_dim1,
                                                shared_hidden_dim2, state_hidden_dim, action_hidden_dim)
        self.target_network = BranchingQNetwork(observation_space, action_space, action_bins, shared_hidden_dim1,
                                                shared_hidden_dim2, state_hidden_dim, action_hidden_dim)
        self.target_network.load_state_dict(self.policy_network.state_dict())

        self.optim = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)  # learned policy

        # select device, and put networks on device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(device)
        self.target_network.to(device)
        self.device = device

        self.target_update_freq = target_update_freq
        self.update_counter = 0

        self.td_target = td_target

    def get_action(self, state_tensor):
        x = state_tensor.to(self.device).T  # single action row vector
        with torch.no_grad():
            out = self.policy_network(x).squeeze(0)
            action = torch.argmax(out, dim=1)  # argmax within each branch

        return action.detach().cpu().numpy()  # action.numpy()

    @staticmethod
    def get_current_qval(bdq_network, states, actions):
        qvals = bdq_network(states)
        return qvals.gather(2, actions.unsqueeze(2)).squeeze(-1)

    def get_next_double_qval(self, next_states):
        # double q learning
        with torch.no_grad():
            argmax = torch.argmax(self.policy_network(next_states), dim=2)
            return self.target_network(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)

    def update_target_net(self):
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.update_counter = 0
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def update_policy(self, batch):
        # get converted batch of tensors
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals = self._extract_tensors(batch)

        # Bellman TD update
        current_Q = self.get_current_qval(self.policy_network, batch_states, batch_actions)
        next_Q = self.get_next_double_qval(batch_next_states)

        # get global target across branches, if desired
        if self.td_target:
            with torch.no_grad():
                if self.td_target == "mean":
                    next_Q = next_Q.mean(1, keepdim=True)
                elif self.td_target == "max":
                    next_Q, _ = next_Q.max(1, keepdim=True)
                else:
                    raise ValueError(f'Either "mean" or "max" must be entered to td_target keyword of BranchingDQN.'
                                     f'You entered {self.td_target}')

            # duplicate columns from reduction op, so current_Q size is same as next_Q for loss function
            next_Q = torch.repeat_interleave(next_Q, current_Q.shape[1], 1)  # TODO should this be done with hooks?

        expected_Q = batch_rewards + next_Q * self.gamma * (1 - batch_terminals)  # target

        loss = F.mse_loss(expected_Q, current_Q)  # minimize TD error with Mean-Squared-Error

        self.optim.zero_grad()
        loss.backward()

        print(f'\n\tLoss = {loss.item()}, Learning...')

        # gradient constraints
        # for p in self.policy_network.parameters():
        #     p.grad.data.clamp_(-1., 1.)

        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.gradient_clip_norm)

        # normalize gradients converging at shared network from action branches and value stream
        if self.rescale_shared_grad_factor is not None:
            for layer in self.policy_network.shared_model:
                layer.weight.grad = layer.weight.grad * self.rescale_shared_grad_factor

        self.optim.step()
        self.update_target_net()

        return loss.detach().cpu()

    def import_model(self, model_path: str):
        """
        Loads a serialized/pickled model from Path, and sets it to the policy and target network.
        :param model_path: Path to pickled PyTorch model
        """
        self.policy_network.load_state_dict(torch.load(model_path))
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def _extract_tensors(self, experiences_batch):
        """Format batch of experiences to proper tensor format."""

        # transpose batch of experiences to "Experience" named tuple of 'batches'
        batch = Experience(*zip(*experiences_batch))
        # convert all (S,A,R,S',t)_i tuple list to np.array then into tensors
        s = torch.Tensor(np.array(batch.state))
        a = torch.Tensor(np.array(batch.action)).long()  # action indices, make int64
        s_ = torch.Tensor(np.array(batch.next_state))
        r = torch.Tensor(np.array(batch.reward)).unsqueeze(1)  # 1-dim
        t = torch.Tensor(np.array(batch.terminal)).unsqueeze(1)  # 1-dim

        return s, a, s_, r, t  # tensor tuple


class EpsilonGreedyStrategy:
    """ Epsilon Greedy action selection strategy, with epsilon decay."""

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step, fixed_epsilon: float = None):
        if fixed_epsilon is not None:
            return fixed_epsilon
        return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
