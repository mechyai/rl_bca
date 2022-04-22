"""
Class for defining and managing all components relative to Branching Dueling Q-Network - BDQN
This BDQN will consist of:
- Double Q-learning
- Dueling Q-network
- Branching Q-network
- Experience Replay
"""

import math
import numpy as np
from typing import Union

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


class SequenceReplayMemory:
    """Manages a sequential replay memory, where data is stored as torch tensors."""

    def __init__(self, capacity: int, batch_size: int, sequence_length: Union[int, list], sequence_ts_spacing: int = 1):
        """
        Creates looping, sequential replay memory.
        :param capacity: Max size of replay memory
        :param batch_size: Size of mini-batch sampling. Sample are not returned at least until this many interactions.
        :param sequence_length: Either int for number of sequence_ts_spacing in sequence, or list of prior timesteps
            for variable spacing sequence, where the current (0th timestep) is automatically included.
        :param sequence_ts_spacing: If fixed sequence spacing, this is the timestep between states in the sequence.
            Ex: 4, for 15 min timesteps --> 0:00, 0:15, 0:30, 0:45, 1:00 in sequence would get 0:00 and 1:00
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- Sequence Management --
        # Variable sequence spacing
        if isinstance(sequence_length, list):
            self.sequence_length = sequence_length if sequence_length[0] != 0 else sequence_length[1:]
            self.sequence_index_span = sequence_length[-1] - 1
            self.previous_sequence_indices = sequence_length
        # Fixed sequence spacing
        else:
            self.sequence_length = sequence_length
            self.sequence_ts_spacing = sequence_ts_spacing
            self.sequence_index_span = (sequence_length - 1) * sequence_ts_spacing
            self.previous_sequence_indices = range(self.sequence_length - 1)

        self.current_index = 0
        self.episode_start_index = 0
        self.episode_start_interaction_count = 0

        self.first_sample = True
        self.total_interaction_count = 0
        self.current_interaction_count = 0

        self.current_sampling_index_end = None
        self.current_sampling_index_start = None
        self.previous_sampling_index_end = None
        self.previous_sampling_index_start = None
        self.current_available_indices = None
        self.current_sample_indices = None

        # -- Replay Buffer --
        self.state_memory = None
        self.action_memory = None
        self.next_state_memory = None
        self.reward_memory = None
        self.terminal_memory = None

    def _get_replay_index(self, unwrapped_index):
        """Returns true, wrapped index based on replay capacity"""

        if hasattr(unwrapped_index, '__iter__'):
            # Handle sequence input
            for i, replay_index in enumerate(unwrapped_index):
                unwrapped_index[i] = replay_index % self.capacity
            return unwrapped_index
        else:
            return unwrapped_index % self.capacity

    def _get_available_sample_indices(self):
        """Gets the proper indices to sample from replay - built around sampling sequence and reuse between episodes."""

        # -- Mixed Episodes in Buffer --
        # Limited by number of interactions thus far, until memory is full or fully overwritten
        # Until replay buffer has been filled to capacity first
        if self.current_interaction_count <= self.capacity:
            # -- Current Episode --
            # Note: replay can provide sample from very start of new episode
            # Need more than sequence span of interactions collected since starting new episode
            if self.current_interaction_count >= self.sequence_index_span:
                # Start based on end of last episode
                self.current_sampling_index_start = self.episode_start_index + self.sequence_index_span

                self.current_sampling_index_end = self.current_index
                # Managing wrapping for interpolation
                if self.current_sampling_index_end < self.current_sampling_index_start:
                    self.current_sampling_index_end += self.capacity
            else:
                self.current_sampling_index_start = 0
                self.current_sampling_index_end = -1

            current_sampling_range = self._get_replay_index(
                np.arange(
                    start=self.current_sampling_index_start,
                    stop=self.current_sampling_index_end + 1,
                )
            )

            # -- Previous Episode --
            # Make sure current index outside last sequence available from prior episode
            previous_sequence_limit_index = self.episode_start_index - 1 - self.sequence_index_span
            if previous_sequence_limit_index < 0:
                # Previous sequence limit IS looped
                previous_episode_sequence_limit_condition = self.current_index \
                                                            < self._get_replay_index(previous_sequence_limit_index)
            else:
                # Previous sequence limit NOT looped
                previous_episode_sequence_limit_condition = not \
                    previous_sequence_limit_index < self.current_index <= self.episode_start_index - 1

            if previous_episode_sequence_limit_condition and self.episode_start_interaction_count != 0:
                # Need 1 sequence spacing from current episode
                self.previous_sampling_index_start = self.current_index + self.sequence_index_span + 1

                self.previous_sampling_index_end = self.episode_start_index - 1
                # Managing wrapping for interpolation
                if self.previous_sampling_index_end < self.previous_sampling_index_start:
                    self.previous_sampling_index_end += self.capacity
            else:
                self.previous_sampling_index_start = 0
                self.previous_sampling_index_end = -1

            # Get sampling range from current and prior episode
            previous_sampling_range = self._get_replay_index(
                np.arange(
                    start=self.previous_sampling_index_start,
                    stop=self.previous_sampling_index_end + 1,
                )
            )

            self.current_available_indices = np.append(previous_sampling_range, current_sampling_range)

        # -- Current Episode ONLY in Buffer --
        else:
            # Memory has been 'over-filled', overwritten entire previous episode
            # Sample from entire memory as normal
            self.current_sampling_index_start = 0
            self.current_sampling_index_end = self.capacity - 1

            current_sampling_range = self._get_replay_index(
                np.arange(
                    start=self.current_sampling_index_start,
                    stop=self.current_sampling_index_end + 1,
                )
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
        self.episode_start_index = self.current_index
        self.episode_start_interaction_count = self.total_interaction_count

    def push(self, state, action, next_state, reward, terminal_flag):
        """Save a transition to replay memory."""

        if self.first_sample:
            self.first_sample = False

            if isinstance(action, int):
                # DQN-based
                action_length = 1
            else:
                # BDQ-based
                action_length = len(action)

            # Init replay memory storage
            self.state_memory = torch.zeros([self.capacity, len(state)]).to(self.device)
            self.action_memory = torch.zeros([self.capacity, action_length], dtype=torch.uint8).to(self.device)
            self.next_state_memory = torch.zeros([self.capacity, len(next_state)]).to(self.device)
            self.reward_memory = torch.zeros([self.capacity, 1]).to(self.device)
            self.terminal_memory = torch.zeros([self.capacity, 1], dtype=torch.uint8).to(self.device)

        # Loop through indices based on size of memory
        index = self.total_interaction_count % self.capacity
        self.current_index = index

        if isinstance(action, int):
            # DQN-based
            action = [action]

        self.state_memory[index] = torch.Tensor(state).to(self.device)
        self.action_memory[index] = torch.ByteTensor(action).to(self.device)
        self.next_state_memory[index] = torch.Tensor(next_state).to(self.device)
        self.reward_memory[index] = torch.Tensor([reward]).to(self.device)
        self.terminal_memory[index] = torch.ByteTensor([terminal_flag]).to(self.device)

        self.total_interaction_count += 1
        self.current_interaction_count = self.total_interaction_count - self.episode_start_interaction_count

    def sample(self):
        """Sample a sequence of transitions randomly"""

        # Get appropriate indices to sample from replay
        self.current_sample_indices = self._get_sample_indices()

        # Get full range of sequence indices for each random sample starting point
        sequence_indices = np.expand_dims(self.current_sample_indices, axis=1)
        for prev_timestep in self.previous_sequence_indices:
            prior_sequence = np.expand_dims(sequence_indices.T[0] - prev_timestep, axis=1)
            # Maintain relative order of seq
            sequence_indices = np.concatenate((prior_sequence, sequence_indices), axis=1)
        sequence_indices = torch.tensor(sequence_indices).long().to(self.device)

        # Sequence sampling
        # {RNN input shape: batch size, sequence len, input size}
        state_batch = self.state_memory[sequence_indices]
        next_state_batch = self.next_state_memory[sequence_indices]

        # No sequence sampling needed, end of sequence
        action_batch = self.action_memory[self.current_sample_indices].long().to(self.device)
        reward_batch = self.reward_memory[self.current_sample_indices].to(self.device)
        terminal_batch = self.terminal_memory[self.current_sample_indices].to(self.device)

        return state_batch, action_batch, next_state_batch, reward_batch, terminal_batch

    def get_single_sequence(self):
        """Returns most recent sequence from replay, corresponding to current state."""

        # Get most recent state, remove prior push count update & index offset
        if isinstance(self.sequence_length, list):
            # Variable sequence spacing
            prior_sequence_indices = np.concatenate((self.current_index - np.array(self.sequence_length),
                                                     np.array([self.current_index])), device=self.device)
        else:
            # Fixed sequence spacing
            start_index = self.current_index - self.sequence_index_span
            prior_sequence_indices = torch.arange(start_index, self.current_index + 1, self.sequence_ts_spacing,
                                                  device=self.device)

        prior_sequence_indices = self._get_replay_index(prior_sequence_indices)

        # RNN input shape: batch size, sequence len, input size
        return self.state_memory[prior_sequence_indices].unsqueeze(0)

    def can_provide_sample(self):
        """Check if replay memory has enough experience tuples to sample batch from."""

        # Such that n sequences of span k can be sampled from batch
        return self.total_interaction_count >= self.batch_size + self.sequence_index_span


class PrioritizedSequenceReplayMemory(SequenceReplayMemory):
    def __init__(self, capacity: int, batch_size: int, sequence_length: int, sequence_ts_spacing: int = 1):
        super().__init__(capacity, batch_size, sequence_length, sequence_ts_spacing)

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

            if self.first_sample:
                self.first_sample = False

                if isinstance(action, int):
                    # DQN-based
                    action_length = 1
                else:
                    # BDQ-based
                    action_length = len(action)

            # Replay Memory
            self.state_memory = torch.zeros([self.capacity, len(state)]).to(self.device)
            self.action_memory = torch.zeros([self.capacity, len(action)], dtype=torch.uint8).to(self.device)
            self.next_state_memory = torch.zeros([self.capacity, len(next_state)]).to(self.device)
            self.reward_memory = torch.zeros([self.capacity, 1]).to(self.device)
            self.terminal_memory = torch.zeros([self.capacity, 1], dtype=torch.uint8).to(self.device)

            # Prioritization
            self.priorities_memory = torch.zeros([self.capacity]).to(self.device)
            self.weights_memory = torch.zeros([self.capacity]).to(self.device)
            self.loss_memory = torch.ones([self.capacity]).to(self.device)

        # Loop through indices based on size of memory
        index = self.total_interaction_count % self.capacity
        self.current_index = index

        if isinstance(action, int):
            # DQN-based
            action = [action]

        # Update replay memory
        self.state_memory[index] = torch.Tensor(state).to(self.device)
        self.action_memory[index] = torch.ByteTensor(action).to(self.device)
        self.next_state_memory[index] = torch.Tensor(next_state).to(self.device)
        self.reward_memory[index] = torch.Tensor([reward]).to(self.device)
        self.terminal_memory[index] = torch.ByteTensor([terminal_flag]).to(self.device)

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
        """Sample a sequence of transitions with PER probabilities."""

        # return super(SequenceReplayMemory).sample(), self.current_sample_indices
        return SequenceReplayMemory.sample(self), self.current_sample_indices


class BranchingQNetwork_RNN(nn.Module):
    """BDQ network architecture with recurrent node."""

    def __init__(self, observation_dim, rnn_hidden_size, rnn_num_layers, action_branches, action_dim,
                 shared_network_size, value_stream_size, advantage_streams_size, lstm=False):
        """
        Below, we define the BDQN architecture's network segments, consisting of a MLP shared representation,
        then a dueling state value module and individual advantage branches. At the end, the value and advantage streams
        are combined to get the branched Q-value output.
        """

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # -- RNN Head --
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        if not lstm:
            self.rnn = nn.GRU(observation_dim, rnn_hidden_size, rnn_num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(observation_dim, rnn_hidden_size, rnn_num_layers, batch_first=True)

        # -- Shared State Feature Estimator --
        layers = []
        prev_layer_size = rnn_hidden_size
        for i, layer_size in enumerate(shared_network_size):
            if layer_size != 0:
                layers.append(nn.Linear(prev_layer_size, layer_size))
                # layers.append(nn.ReLU())
                prev_layer_size = layer_size

        shared_final_layer = prev_layer_size
        self.shared_model = nn.Sequential(*layers)

        # --- Value Stream ---
        layers = []
        prev_layer_size = shared_final_layer
        for i, layer_size in enumerate(value_stream_size):
            if layer_size != 0:
                layers.append(nn.Linear(prev_layer_size, layer_size))
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
                    prev_layer_size = layer_size
            final_layer = nn.Linear(prev_layer_size, action_dim)
            self.advantage_streams.append(nn.Sequential(*layers, final_layer))

    def forward(self, state_input):
        # RNN Node (num layers, batch size, hidden size)
        # Hidden (h0 and c0 for LSTM) is automatically 0 if not included
        # h0 = torch.zeros(self.rnn_num_layers, state_input.size(0), self.rnn_hidden_size).to(self.device)

        out, _ = self.rnn(state_input)  # out: batch size, seq len, hidden size
        out = out[:, -1, :]  # get last timestep output (many to one)
        # Shared Network
        if len(self.shared_model) != 0:
            out = self.shared_model(out)
        # Value Branch
        value = self.value_stream(out)
        # Advantage Streams
        advs = torch.stack([advantage_stream(out) for advantage_stream in self.advantage_streams], dim=1)

        # Q-Value - Recombine Branches
        q_vals = value.unsqueeze(2) + advs - advs.mean(2, keepdim=True)  # identifiable method eqn #1

        return q_vals


class BranchingDQN_RNN(nn.Module):
    """A branching, dueling, & double DQN algorithm."""

    def __init__(self, observation_dim: int, rnn_hidden_size: int, rnn_num_layers: int, action_branches: int,
                 action_dim: int, shared_network_size: list, value_stream_size: list, advantage_streams_size: list,
                 target_update_freq: int, learning_rate: float, gamma: float, td_target: str,
                 gradient_clip_norm: float, rescale_shared_grad_factor: float = None, lstm: bool = False,
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

        self.policy_network = BranchingQNetwork_RNN(observation_dim, rnn_hidden_size, rnn_num_layers, action_branches,
                                                    action_dim, shared_network_size, value_stream_size,
                                                    advantage_streams_size, lstm=lstm)
        self.target_network = BranchingQNetwork_RNN(observation_dim, rnn_hidden_size, rnn_num_layers, action_branches,
                                                    action_dim, shared_network_size, value_stream_size,
                                                    advantage_streams_size, lstm=lstm)
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
        """Get greedy action from current state and past sequence included."""

        x = state_tensor.to(self.device)  # single action row vector
        with torch.no_grad():
            out = self.policy_network(x).squeeze(0)
            action = torch.argmax(out, dim=1)  # argmax within each branch

        return action.detach().cpu().numpy()  # action.numpy()

    def update_learning_rate(self):
        """Based on conditions, the optimizer's learning rate is dynamically updated."""
        lr = 0.004
        if False:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    @staticmethod
    def get_current_qval(bdq_network, states, actions):
        """For given state and action, get the associated Q-val from the network output."""

        qvals = bdq_network(states)
        return qvals.gather(2, actions.unsqueeze(2)).squeeze(-1)

    def get_next_double_qval(self, next_states):
        """Get next Q-value for a given next-state, via double q-learning method."""

        # double q learning
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
        for target_param, learned_param in zip(self.target_network.parameters(), self.policy_network.parameters()):
            target_param.data.copy_(tau * learned_param.data + (1.0 - tau) * target_param.data)

    def update_policy(self, batch, gradient_weights=None):
        """Learn from batch of interaction tuples. Optimizes learned policy DQN."""

        # Get converted batch of tensors
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
            next_Q = torch.repeat_interleave(next_Q, current_Q.shape[1], 1)  # TODO should this be done with hooks?

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

        self.optimizer.zero_grad()
        loss_total.backward()

        # -- Modify Gradients --
        if self.gradient_clip_norm != 0:
            # If 0, don't clip norm
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.gradient_clip_norm)

        # Normalize gradients converging at shared network from action branches and value stream
        if self.rescale_shared_grad_factor is not None:
            for layer in self.policy_network.shared_model:
                layer.weight.grad = layer.weight.grad * self.rescale_shared_grad_factor

        # -- Optimize --
        self.optimizer.step()

        # -- Update --
        if self.target_update_freq > 1:
            self.target_hard_update()
        else:
            self.target_soft_update()

        self.step_count += 1

        return float(loss_total.detach().cpu()), loss_each.detach().mean(dim=1)

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
