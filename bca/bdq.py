"""
Class for defining and managing all components relative to Branching Dueling Q-Network - BDQN
This BDQN will consist of:
- Double Q-learning
- Dueling Q-network
- Branching Q-network
- Experience Replay
"""

import math
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

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
    """Manages a replay memory, where data is stored as torch tensors."""

    def __init__(self, capacity, batch_size):
        """
        Creates looping replay memory.
        :param capacity: Max size of replay memory
        :param batch_size: Size of mini-batch sampling. Sample are not returned at least until this many interactions.
        """

        self.capacity = capacity
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.current_index = 0
        self.episode_start_interaction_count = 0

        self.first_sample = True
        self.total_interaction_count = 0
        self.current_interaction_count = 0


        self.current_sampling_index_end = None
        self.current_sampling_index_start = None
        self.current_available_indices = None
        self.current_sample_indices = None

        # -- Replay Buffer --
        self.state_memory = None
        self.action_memory = None
        self.next_state_memory = None
        self.reward_memory = None
        self.terminal_memory = None

    def _get_available_sample_indices(self):
        """Gets the proper indices to sample from replay - built around sampling sequence and reuse between episodes."""

        if self.total_interaction_count < self.capacity:
            # Replay buffer has not be 'over-filled yet'
            self.current_sampling_index_start = 0
            self.current_sampling_index_end = self.current_index
        else:
            # Replay buffer has been 'over-filled'
            self.current_sampling_index_start = 0
            self.current_sampling_index_end = self.capacity - 1

        current_sampling_range = np.arange(
                start=self.current_sampling_index_start,
                stop=self.current_sampling_index_end + 1
        )

        self.current_available_indices = current_sampling_range

    def _get_sample_indices(self):
        # Update available sample indices from replay
        self._get_available_sample_indices()

        # Get sampling indices
        sample_indices = np.random.choice(
            a=self.current_available_indices,
            size=self.batch_size,
            replace=False
        )

        return sample_indices

    def reset_between_episode(self):
        """Manages resetting of specific attributes to manage the replay between episodes."""

        self.current_interaction_count = 0
        self.episode_start_interaction_count = self.total_interaction_count

    def push(self, state, action, next_state, reward, terminal_flag):
        """Save a transition to replay memory"""

        if self.first_sample:
            self.first_sample = False

            # Init replay memory storage
            self.state_memory = torch.zeros([self.capacity, len(state)]).to(self.device)
            self.action_memory = torch.zeros([self.capacity, len(action)]).to(self.device)
            self.next_state_memory = torch.zeros([self.capacity, len(next_state)]).to(self.device)
            self.reward_memory = torch.zeros([self.capacity, 1]).to(self.device)
            self.terminal_memory = torch.zeros([self.capacity, 1]).to(self.device)

        # Loop through indices based on size of memory
        index = self.total_interaction_count % self.capacity
        self.current_index = index

        self.state_memory[index] = torch.Tensor(state).to(self.device)
        self.action_memory[index] = torch.Tensor(action).to(self.device)
        self.next_state_memory[index] = torch.Tensor(next_state).to(self.device)
        self.reward_memory[index] = torch.Tensor([reward]).to(self.device)
        self.terminal_memory[index] = torch.Tensor([terminal_flag]).to(self.device)

        self.total_interaction_count += 1
        self.current_interaction_count = self.total_interaction_count - self.episode_start_interaction_count

    def sample(self):
        """Sample transitions with random probability."""

        # Get appropriate indices to sample from replay
        self.current_sample_indices = self._get_sample_indices()
        sample_indices = self.current_sample_indices

        state_batch = self.state_memory[sample_indices].to(self.device)
        action_batch = self.action_memory[sample_indices].long().to(self.device)
        next_state_batch = self.next_state_memory[sample_indices].to(self.device)
        reward_batch = self.reward_memory[sample_indices].to(self.device)
        terminal_batch = self.terminal_memory[sample_indices].to(self.device)

        return state_batch, action_batch, next_state_batch, reward_batch, terminal_batch

    def can_provide_sample(self):
        """Check if replay memory has enough experience tuples to sample batch from"""

        return self.total_interaction_count >= self.batch_size


class PrioritizedReplayMemory(ReplayMemory):
    """Manages a prioritized replay memory, where data is stored as torch tensors."""

    def __init__(self, capacity: int, batch_size):
        super().__init__(capacity, batch_size)

        # PER
        self.loss_memory = None
        self.weights_memory = None
        self.priorities_memory = None

        self.max_loss = 0.0001
        self.alpha = 1
        self.betta = 1

    def push(self, state, action, next_state, reward, terminal_flag):
        """Save a transition to replay memory"""

        if self.first_sample:
            self.first_sample = False

            # Replay Memory
            self.state_memory = torch.zeros([self.capacity, len(state)]).to(self.device)
            self.action_memory = torch.zeros([self.capacity, len(action)]).to(self.device)
            self.next_state_memory = torch.zeros([self.capacity, len(next_state)]).to(self.device)
            self.reward_memory = torch.zeros([self.capacity, 1]).to(self.device)
            self.terminal_memory = torch.zeros([self.capacity, 1]).to(self.device)

            # Prioritization
            self.priorities_memory = torch.zeros([self.capacity]).to(self.device)
            self.weights_memory = torch.zeros([self.capacity]).to(self.device)
            self.loss_memory = torch.ones([self.capacity]).to(self.device)

        # Loop through indices based on size of memory
        index = self.total_interaction_count % self.capacity
        self.current_index = index

        # Update replay memory
        self.state_memory[index] = torch.Tensor(state).to(self.device)
        self.action_memory[index] = torch.Tensor(action).to(self.device)
        self.next_state_memory[index] = torch.Tensor(next_state).to(self.device)
        self.reward_memory[index] = torch.Tensor([reward]).to(self.device)
        self.terminal_memory[index] = torch.Tensor([terminal_flag]).to(self.device)

        # Update priorities
        self.loss_memory[index] = self.max_loss

        self.total_interaction_count += 1
        self.current_interaction_count = self.total_interaction_count - self.episode_start_interaction_count

    def update_td_losses(self, sample_indices, loss_each):
        """Update the tracked losses (TD error) of the batch."""

        # Update max priority for added samples
        max_loss = max(loss_each)  # Doing Max over batch to be quicker, instead of whole buffer everytime
        if max_loss > self.max_loss:
            self.max_loss = max_loss

        # Add new loss values to memory
        self.loss_memory[sample_indices] = loss_each

    def get_gradient_weights(self, sample_indices, betta=None):
        """Return weights for given sample indices from batch for backpropagation to remove bias form priorities."""

        if betta is None:
            betta = self.betta

        # Current number of available memory
        n = len(self.current_available_indices)
        self.weights_memory[self.current_available_indices] = torch.pow(
            (1 / self.priorities_memory[self.current_available_indices]) / n,
            betta
        )

        return self.weights_memory[sample_indices]

    def get_priority_probabilities(self, alpha=None):
        """Convert scalar probability values per sample to probability distribution = 100%"""

        if alpha is None:
            alpha = self.alpha

        # Get priorities from TD loss proxy
        losses = self.loss_memory[self.current_available_indices]
        sum_priorities = torch.pow(losses, alpha).sum()

        self.priorities_memory[self.current_available_indices] = torch.pow(losses, alpha) / sum_priorities

        return self.priorities_memory[self.current_available_indices]

    def _get_sample_indices(self):
        # Update available sample indices from replay
        self._get_available_sample_indices()
        probabilities = np.array(self.get_priority_probabilities().cpu())

        # Get sampling indices
        sample_indices = np.random.choice(
            a=self.current_available_indices,
            size=self.batch_size,
            replace=False,
            p=probabilities
        )

        return sample_indices

    def sample(self):
        """Sample transitions with weighted priority probabilities. Return batch with each index of interaction."""

        # return super(ReplayMemory).sample(), self.current_sample_indices
        return ReplayMemory.sample(self), self.current_sample_indices


class BranchingQNetwork(nn.Module):
    """BDQ network architecture."""

    def __init__(self, observation_dim, action_branches, action_dim,
                 shared_network_size, value_stream_size, advantage_streams_size):
        """
        Below, we define the BDQN architecture's network segments, consisting of RNN node & MLP shared representation,
        then a dueling state value module and individual advantage branches. At the end, the value and advantage streams
        are combined to get the branched Q-value output.
        """

        super().__init__()

        # -- Shared State Feature Estimator --
        layers = []
        prev_layer_size = observation_dim
        for i, layer_size in enumerate(shared_network_size):
            if layer_size != 0:
                layers.append(nn.Linear(prev_layer_size, layer_size))
                layers.append(nn.ReLU())
                prev_layer_size = layer_size

        shared_final_layer = prev_layer_size
        self.shared_model = nn.Sequential(*layers)

        # --- Value Stream ---
        layers = []
        prev_layer_size = shared_final_layer
        for i, layer_size in enumerate(value_stream_size):
            if layer_size != 0:
                layers.append(nn.Linear(prev_layer_size, layer_size))
                layers.append(nn.ReLU())
                prev_layer_size = layer_size

        final_layer = nn.Linear(prev_layer_size, 1)  # output state-value
        self.value_stream = nn.Sequential(*layers, final_layer)

        # --- Advantage Streams ---
        self.advantage_streams = nn.ModuleList()
        for branch in range(action_branches):
            layers = []
            prev_layer_size = shared_final_layer
            for i, layer_size in enumerate(advantage_streams_size):
                if layer_size != 0:
                    layers.append(nn.Linear(prev_layer_size, layer_size))
                    layers.append(nn.ReLU())
                    prev_layer_size = layer_size
            final_layer = nn.Linear(prev_layer_size, action_dim)
            self.advantage_streams.append(nn.Sequential(*layers, final_layer))

    def forward(self, state_input):
        # Shared Network
        out = self.shared_model(state_input)
        # Value Branch
        value = self.value_stream(out)
        # Advantage Streams
        advs = torch.stack([advantage_stream(out) for advantage_stream in self.advantage_streams], dim=1)

        # Q-Value - Recombine Branches
        q_vals = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)  # identifiable method eqn #1

        return q_vals


class BranchingDQN(nn.Module):
    """A branching, dueling, & double DQN algorithm."""

    def __init__(self, observation_dim: int, action_branches: int, action_dim: int,
                 shared_network_size: list, value_stream_size: list, advantage_streams_size: list,
                 target_update_freq: int, learning_rate: float, gamma: float, td_target: str,
                 gradient_clip_norm: float, rescale_shared_grad_factor: float = None,
                 optimizer: str = 'Adam', **optimizer_kwargs):

        super().__init__()

        self.observation_space = observation_dim
        self.action_branches = action_branches
        self.action_dim = action_dim
        self.shared_network_size = shared_network_size
        self.advantage_streams_size = advantage_streams_size
        self.value_stream_size = value_stream_size
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm
        self.rescale_shared_grad_factor = rescale_shared_grad_factor

        self.policy_network = BranchingQNetwork(observation_dim, action_branches, action_dim, shared_network_size,
                                                value_stream_size, advantage_streams_size)
        self.target_network = BranchingQNetwork(observation_dim, action_branches, action_dim, shared_network_size,
                                                value_stream_size, advantage_streams_size)
        self.target_network.load_state_dict(self.policy_network.state_dict())  # copy params

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(self.device)
        self.target_network.to(self.device)

        # self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)  # learned policy
        self.optimizer = \
            getattr(optim, optimizer)(self.policy_network.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.step_count = 0

        self.td_target = td_target

    def get_greedy_action(self, state_tensor):
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
        # Double q learning implementation
        with torch.no_grad():
            argmax = torch.argmax(self.policy_network(next_states), dim=2)
            return self.target_network(next_states).gather(2, argmax.unsqueeze(2)).squeeze(-1)

    def target_hard_update(self):
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_count = 0
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def target_soft_update(self):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        tau = 0.01
        self.update_count += 1
        for target_param, learned_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(tau * learned_param.data + (1.0 - tau) * target_param.data)

    def update_policy(self, batch, gradient_weights=None):
        # get converted batch of tensors
        # batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals = self._extract_tensors(batch)
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals = batch

        # Bellman TD update
        current_Q = self.get_current_qval(self.policy_network, batch_states, batch_actions)
        next_Q = self.get_next_double_qval(batch_next_states)

        # Get global target across branches, if desired
        if self.td_target:
            with torch.no_grad():
                if self.td_target == 'mean':  # mean
                    next_Q = next_Q.mean(1, keepdim=True)
                elif self.td_target == 'max':  # max
                    next_Q, _ = next_Q.max(1, keepdim=True)
                else:
                    raise ValueError(f'Either "mean" or "max" must be entered to td_target keyword of BranchingDQN.'
                                     f'You entered {self.td_target}')

            # duplicate columns from reduction op, so current_Q size is same as next_Q for loss function
            next_Q = torch.repeat_interleave(next_Q, current_Q.shape[1], 1)

        expected_Q = batch_rewards + next_Q * self.gamma * (1 - batch_terminals)  # target

        # loss_old = F.mse_loss(expected_Q, current_Q)  # minimize TD error with Mean-Squared-Error

        loss_fn = nn.MSELoss(reduction='none')  # Get no mean reduction
        loss_each = loss_fn(expected_Q, current_Q)  # Capture each individual loss component

        # For PER
        if gradient_weights is not None:
            # Re-weight gradients through each samples loss
            loss_total = torch.mean(loss_each * gradient_weights.unsqueeze(dim=1))
        else:
            loss_total = torch.mean(loss_each)

        # Backpropagation
        self.optimizer.zero_grad()
        loss_total.backward()

        # -- Modify Gradients --
        if self.gradient_clip_norm != 0:
            # If 0, don't clip norm
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.gradient_clip_norm)

        # Normalize gradients converging at shared network from action branches and value stream
        if self.rescale_shared_grad_factor is not None:
            for layer in self.policy_network.shared_model:
                if hasattr(layer, 'weight'):  # Ignore activation layers
                    layer.weight.grad = layer.weight.grad * self.rescale_shared_grad_factor

        # -- Optimize --
        self.optimizer.step()

        # -- Update --
        if self.target_update_freq > 1:
            self.target_hard_update()
        else:
            self.target_soft_update()

        self.step_count += 1

        return float(loss_total.detach().cpu()), loss_each.detach().mean(dim=1).cpu()

    def import_model(self, model_path: str):
        """
        Loads a serialized/pickled model from Path, and sets it to the policy and target network.
        :param model_path: Path to pickled PyTorch model
        """
        self.policy_network.load_state_dict(torch.load(model_path))
        # self.policy_network.load_state_dict(torch.load(model_path).state_dict())  # used to save wrong
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def change_learning_rate_discrete(self, lr: float, **optimizer_kwargs):
        """For predefined network, change the learning rate of the given optimizer."""

        # Get optimizer in use
        optimizer = self.optimizer.__class__.__name__
        self.optimizer = \
            getattr(optim, optimizer)(self.policy_network.parameters(), lr=lr, **optimizer_kwargs)

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
        self.current_exploration_rate = 0

    def get_exploration_rate(self, current_step, fixed_epsilon: float = None):
        if fixed_epsilon is not None:
            self.current_exploration_rate = fixed_epsilon
            return fixed_epsilon
        self.current_exploration_rate = self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)
        return self.current_exploration_rate
