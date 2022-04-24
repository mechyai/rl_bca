"""
Class for defining and managing all components relative to Branching Dueling Q-Network - BDQN
This BDQN will consist of:
- Double Q-learning
- Dueling Q-network
- Branching Q-network
- Experience Replay
"""

import math

import torch
import torch.nn as nn
import torch.optim as optim

""" 
DQN:
Guides-
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://deeplizard.com/learn/video/PyQNfsGUnQA

Dueling DQN:
Guides-
https://www.youtube.com/watch?v=Odmeb3gkN0M&t=3s - Eden Meyer
Repos-
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DuelingQNetwork_RNN(nn.Module):
    """Deep Q-network architecture."""

    def __init__(self, observation_dim, rnn_hidden_size, rnn_num_layers, action_branches, action_dim,
                 shared_network_size, value_stream_size, advantage_streams_size, lstm=False):

        super().__init__()

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
        layers = []
        prev_layer_size = shared_final_layer
        for i, layer_size in enumerate(advantage_streams_size):
            if layer_size != 0:
                layers.append(nn.Linear(prev_layer_size, layer_size))
                layers.append(nn.ReLU())
                prev_layer_size = layer_size

        final_layer = nn.Linear(prev_layer_size, action_dim ** action_branches)
        self.advantage_stream = (nn.Sequential(*layers, final_layer))

    def forward(self, state_input):
        """Get q-values output for given state"""

        # RNN Node (num layers, batch size, hidden size)
        # Hidden (h0 and c0 for LSTM) is automatically 0 if not included
        # h0 = torch.zeros(self.rnn_num_layers, state_input.size(0), self.rnn_hidden_size).to(device)
        out, _ = self.rnn(state_input)  # out: batch size, seq len, hidden size
        out = out[:, -1, :]  # get last timestep output (many to one)
        # Shared Network
        if len(self.shared_model) != 0:
            out = self.shared_model(out)
        # Value Branch
        value = self.value_stream(out)
        # Advantage Streams
        advs = self.advantage_stream(out)

        # Q-Value - Recombine Branches
        q_vals = value.unsqueeze(2) + advs - advs.mean(1, keepdim=True)  # identifiable method eqn #1

        return q_vals


class DuelingDQN_RNN(nn.Module):
    """A DQN algorithm with RNN optional."""

    def __init__(self, observation_dim: int, rnn_hidden_size: int, rnn_num_layers: int, action_branches: int,
                 action_dim: int, shared_network_size: list, value_stream_size: list, advantage_stream_size: list,
                 target_update_freq: int, learning_rate: float, gamma: float, lstm: bool = False,
                 gradient_clip_norm: float = 1, optimizer: str = 'Adam', **optimizer_kwargs):

        super().__init__()

        self.observation_space = observation_dim
        self.action_branches = action_branches
        self.action_dim = action_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers
        self.shared_network_size = shared_network_size
        self.advantage_streams_size = advantage_stream_size
        self.value_stream_size = value_stream_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.gradient_clip_norm = gradient_clip_norm

        self.policy_network = DuelingQNetwork_RNN(observation_dim, rnn_hidden_size, rnn_num_layers, action_branches,
                                                  action_dim, shared_network_size, value_stream_size,
                                                  advantage_stream_size, lstm=lstm)
        self.target_network = DuelingQNetwork_RNN(observation_dim, rnn_hidden_size, rnn_num_layers, action_branches,
                                                  action_dim, shared_network_size, value_stream_size,
                                                  advantage_stream_size, lstm=lstm)
        self.target_network.load_state_dict(self.policy_network.state_dict())  # copy params

        self.policy_network.to(device)
        self.target_network.to(device)

        # self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)  # learned policy
        self.optimizer = \
            getattr(optim, optimizer)(self.policy_network.parameters(), lr=learning_rate, **optimizer_kwargs)

        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.step_count = 0

    def get_greedy_action(self, state_tensor):
        x = state_tensor.to(device).T  # single action row vector

        with torch.no_grad():
            q_value = self.policy_network(x).squeeze(0)
            action = torch.argmax(q_value)  # argmax within each branch

        return int(action.detach().cpu())  # action.numpy()

    @staticmethod
    def get_current_qval(dqn_network, states, actions):
        qvals = dqn_network(states)
        return qvals.gather(1, actions)

    def get_next_double_qval(self, next_states):
        # Double q learning implementation
        with torch.no_grad():
            action_argmax = torch.argmax(self.policy_network(next_states), dim=1)
            return self.target_network(next_states).gather(1, action_argmax.unsqueeze(1))

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
        # Get batch of tensors
        batch_states, batch_actions, batch_next_states, batch_rewards, batch_terminals = batch

        # Bellman TD update
        current_Q = self.get_current_qval(self.policy_network, batch_states, batch_actions)
        next_Q = self.get_next_double_qval(batch_next_states)

        target_Q = batch_rewards + next_Q * self.gamma * (1 - batch_terminals)  # target

        loss_fn = nn.MSELoss(reduction='none')  # Get no mean reduction
        loss_each = loss_fn(target_Q, current_Q)  # Capture each individual loss component

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

        # Normalize gradients converging at shared network from advantage stream and value stream
        for layer in self.policy_network.shared_model:
            if hasattr(layer, 'weight'):  # Ignore activation layers
                layer.weight.grad = layer.weight.grad * 1 / 2

        # -- Optimize --
        self.optimizer.step()

        # -- Target Update --
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
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def change_learning_rate_discrete(self, lr: float, **optimizer_kwargs):
        """For predefined network, change the learning rate of the given optimizer."""

        # Get optimizer in use
        optimizer = self.optimizer.__class__.__name__
        self.optimizer = \
            getattr(optim, optimizer)(self.policy_network.parameters(), lr=lr, **optimizer_kwargs)
