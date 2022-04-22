import copy
import random
from collections import namedtuple
from itertools import product

from emspy import BcaEnv, MdpManager

from bca import Agent
from bca import BranchingDQN, ReplayMemory, PrioritizedReplayMemory, EpsilonGreedyStrategy
from bca import BranchingDQN_RNN, SequenceReplayMemory, PrioritizedSequenceReplayMemory
from bca import DQN, DQN_RNN

from bca_manager import ModelManager, TensorboardManager


class RunManager:
    """This class helps manage all hparams and sampling, as well as creating all objects dependent on these hparams."""
    # -- Agent Params --
    # Misc. Params
    action_branches = 4

    selected_params = {
        # -- Agent Params --
        'observation_ts_frequency': 5,
        'actuation_ts_frequency': 5,
        'learning_loops': 10,

        # --- Behavioral Policy ---
        'eps_start': 0.15,
        'eps_end': 0.001,
        'eps_decay': 1e-6,

        # --- Experience Replay ---
        'replay_capacity': 64,
        'batch_size': 8,

        # DQN or BDQ
        'model': 1,  # 1=DQN, 2=Dueling DQN, 3=BDQ
        # PER
        'PER': True,
        # RNN
        'rnn': True,

        # -- BDQ --
        # Fixed
        'observation_dim': 51,
        'action_branches': action_branches,  # n building zones
        'actuation_function': 7,

        # TD Update
        'optimizer': 'Adagrad',
        'learning_rate': 5e-4,
        'gamma': 0.8,

        # Reward
        'reward_aggregation': 'sum',  # sum or mean
        'reward_sparsity_ts': 1,
        'reward_scale': 0.1,
        'reward_clipping': 0,
        'lambda_rtp': 0.01,

        # Network mods
        'td_target': 'mean',  # (0) mean or (1) max
        'gradient_clip_norm': 5,  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': 1 / (action_branches),
        'target_update_freq': 0.05,  # [50, 150, 500, 1e3, 1e4],  # consider n learning loops too
    }
    if selected_params['model'] == 3:
        # BDQ-based
        architecture_params = {
            'shared_network_size': [124, 64],
            'value_stream_size': [64, 32],
            'advantage_streams_size': [32, 16]
        }
        selected_params = {**selected_params, **architecture_params}

    if selected_params['model'] == 1 or selected_params['model'] == 2:
        # DQN-based
        architecture_params = {
            'network_size': [124, 124, 64]
        }
        selected_params = {**selected_params, **architecture_params}

    if selected_params['rnn']:
        rnn_params = {
            # -- State Sequence --
            'sequence_ts_spacing': 6,
            'sequence_length': 4,  # input as list for variable ts spacing

            # -- BDQ Architecture --
            'lstm': True,
            'rnn_hidden_size': 32,
            'rnn_num_layers': 3,
        }
        selected_params = {**selected_params, **rnn_params}

    if selected_params['PER']:
        per_params = {
            'alpha_start': 1,
            'alpha_decay_factor': None,
            'betta_start': 0.5,
            'betta_decay_factor': 1e-5,
        }
        selected_params = {**selected_params, **per_params}

    Run = namedtuple('Run', selected_params.keys())
    selected_params = Run(*selected_params.values())

    # ----- For Hyperparameter Search -----
    hyperparameter_dict = {
        # -- Agent Params --
        'observation_ts_frequency': [5],  # * [5, 10, 15],
        'actuation_ts_frequency': [5],  # * [5, 10, 15],
        'learning_loops': [5],

        # --- Behavioral Policy ---
        'eps_start': [0.2, 0.05],
        'eps_end': [0.001],
        'eps_decay': [1e-4],

        # --- Experience Replay ---
        'replay_capacity': [500, 2000],
        'batch_size': [8, 32, 96],

        # PER
        'PER': [False],
        # RNN
        'rnn': [True],

        # -- BDQ --
        # Fixed
        'observation_dim': [61],
        'action_branches': [action_branches],  # n building zones
        'actuation_function': [5],  # ----------------------------------------------------------------------------------

        # Architecture
        'shared_network_size_l1': [96],
        'shared_network_size_l2': [96],
        'value_stream_size_l1': [64],
        'value_stream_size_l2': [64],
        'advantage_streams_size_l1': [48],
        'advantage_streams_size_l2': [0],

        # TD Update
        'optimizer': ['Adagrad'],
        'learning_rate': [5e-4],
        'gamma': [0.8],

        # Reward
        'reward_aggregation': ['sum'],  # sum or mean
        'reward_sparsity_ts': [1],
        'lambda_rtp': [0.3 * 3],
        'reward_scale': [0.01],

        # Network mods
        'td_target': ['mean'],  # (0) mean or (1) max
        'gradient_clip_norm': [2],  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': [1 / (action_branches)],
        'target_update_freq': [5e2, 5e3],  # [50, 150, 500, 1e3, 1e4],  # consider n learning loops too
    }

    if True in hyperparameter_dict['rnn']:
        rnn_params = {
            # -- State Sequence --
            'sequence_ts_spacing': [1, 6],
            'sequence_length': [6, 12],

            # -- BDQ Architecture --
            'lstm': [True],
            'rnn_hidden_size': [48, 96],
            'rnn_num_layers': [1, 2],
        }
        hyperparameter_dict = {**hyperparameter_dict, **rnn_params}

    if True in hyperparameter_dict['PER']:
        per_params = {
            'alpha_start': [1],
            'alpha_decay_factor': [None],
            'betta_start': [0.4],
            'betta_decay_factor': [1e-5],
        }
        hyperparameter_dict = {**hyperparameter_dict, **per_params}

    def __init__(self):
        self.mdp = None
        self.agent = None
        self.experience_replay = None
        self.policy = None
        self.dqn = None

        self.runs = self.get_runs(self.hyperparameter_dict)
        self.n_runs = len(self.runs)

    @staticmethod
    def get_runs(params: dict):
        """Get all permutations of hyperparameters passed."""

        Run = namedtuple('Run', params.keys())

        runs = []
        for config in product(*params.values()):
            run_config = Run(*config)
            # Handle incompatible hyperparam configuration
            if run_config.batch_size > run_config.replay_capacity:
                pass
            else:
                runs.append(run_config)

        return runs

    def get_runs_modified(self, modified_params_dict: dict):
        """Get all permutations of hyperparameters passed. Passed dict must contain only keys aligned with Run tuple."""

        selected_params = copy.copy(self.selected_params)
        # Add custom changed values to param dict
        for param_name, values in modified_params_dict.items():
            selected_params[param_name] = values

        # Make sure all values in dict are of type list
        for param_name, values in selected_params.items():
            if not isinstance(values, list):
                selected_params[param_name] = [values]  # make list

        return self.get_runs(selected_params)

    def shuffle_runs(self, runs: None):
        """Returns list of shuffled runs"""

        if runs is None:
            runs = self.runs

        runs = copy.copy(runs)
        random.shuffle(runs)

        return runs

    def create_agent(self, run, mdp: MdpManager, sim: BcaEnv, model: ModelManager,
                     tensorboard_manager: TensorboardManager, current_step: int = 0, continued_parameters: dict = None,
                     print_values: bool = False):
        """Creates and returns new RL Agent from defined parameters."""

        self.agent = Agent(
            emspy_sim=sim,
            mdp=mdp,
            bem_model=model,
            dqn_model=self.dqn,
            policy=self.policy,
            replay_memory=self.experience_replay,
            run_parameters=run,
            rnn=run.rnn,
            observation_frequency=run.observation_ts_frequency,
            actuation_frequency=run.actuation_ts_frequency,
            actuation_dimension=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
            reward_aggregation=run.reward_aggregation,
            learning_loop=run.learning_loops,
            tensorboard_manager=tensorboard_manager,
            current_step=current_step,
            continued_parameters=continued_parameters,
            print_values=print_values
        )

        return self.agent

    def create_policy(self, run: namedtuple):
        """Creates and returns new RL policy from defined parameters."""

        self.policy = EpsilonGreedyStrategy(
            start=run.eps_start,
            end=run.eps_end,
            decay=run.eps_decay
        )

        return self.policy

    def create_replay_memory(self, run: namedtuple):
        """Creates and returns new Experience Replay from defined parameters."""

        if run.rnn:
            if run.PER:
                self.experience_replay = PrioritizedSequenceReplayMemory(
                    capacity=run.replay_capacity,
                    batch_size=run.batch_size,
                    sequence_length=run.sequence_length,
                    sequence_ts_spacing=run.sequence_ts_spacing,
                )
            else:
                self.experience_replay = SequenceReplayMemory(
                    capacity=run.replay_capacity,
                    batch_size=run.batch_size,
                    sequence_length=run.sequence_length,
                    sequence_ts_spacing=run.sequence_ts_spacing
                )

        elif run.PER:
            self.experience_replay = PrioritizedReplayMemory(
                capacity=run.replay_capacity,
                batch_size=run.batch_size,
            )
        else:
            self.experience_replay = ReplayMemory(
                capacity=run.replay_capacity,
                batch_size=run.batch_size
            )

        return self.experience_replay

    def create_bdq(self, run: namedtuple):
        """Creates and returns new BDQ model from defined parameters."""

        # DQN
        if run.model == 1:
            if run.rnn:
                self.dqn = DQN_RNN(
                    observation_dim=run.observation_dim,
                    action_branches=run.action_branches,
                    rnn_hidden_size=run.rnn_hidden_size,
                    rnn_num_layers=run.rnn_num_layers,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    network_size=run.network_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor,
                    lstm=run.lstm
                )
            else:
                self.dqn = DQN(
                    observation_dim=run.observation_dim,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    network_size=run.network_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor,
                )

        # Dueling DQN
        elif run.model == 2:
            pass
        # BDQ
        elif run.model == 3:
            if run.rnn:
                self.dqn = BranchingDQN_RNN(
                    observation_dim=run.observation_dim,
                    rnn_hidden_size=run.rnn_hidden_size,
                    rnn_num_layers=run.rnn_num_layers,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.shared_network_size,
                    value_stream_size=run.value_stream_size,
                    advantage_streams_size=run.advantage_streams_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    td_target=run.td_target,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor,
                    lstm=run.lstm
                )
            else:
                self.dqn = BranchingDQN(
                    observation_dim=run.observation_dim,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.shared_network_size,
                    value_stream_size=run.value_stream_size,
                    advantage_streams_size=run.advantage_streams_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    td_target=run.td_target,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor
                )

        return self.dqn
