import copy
import random
from collections import namedtuple
from itertools import product

from emspy import BcaEnv, MdpManager

from bca import Agent
from bca import BranchingDQN, ReplayMemory, PrioritizedReplayMemory, EpsilonGreedyStrategy
from bca import BranchingDQN_RNN, SequenceReplayMemory, PrioritizedSequenceReplayMemory
from bca import DQN, DQN_RNN
from bca import DuelingDQN, DuelingDQN_RNN

from bca_manager import ModelManager, TensorboardManager


class RunManager:
    """This class helps manage all hparams and sampling, as well as creating all objects dependent on these hparams."""
    # -- Agent Params --
    # Misc. Params
    action_branches = 1

    selected_params = {
        # -- Agent Params --
        'observation_ts_frequency': 5,
        'actuation_ts_frequency': 5,
        'learning_loops': 10,

        # --- Behavioral Policy ---
        'eps_start': 0.25,
        'eps_end': 0.01,
        'eps_decay': 5e-7,

        # --- Experience Replay ---
        'replay_capacity': 2048,
        'batch_size': 128,

        # DQN or BDQ
        'model': 2,  # 1=DQN, 2=Dueling DQN, 3=BDQ
        # PER
        'PER': False,
        # RNN
        'rnn': True,

        # -- BDQ --
        # Fixed
        'observation_dim': 18,
        'action_branches': action_branches,  # n building zones
        'actuation_function': 5,

        # TD Update
        'optimizer': 'Adagrad',
        'learning_rate': 5e-1,
        'lr_scheduler': 'ReduceLROnPlateau',
        'gamma': 0.9,

        # Reward
        'reward_aggregation': 'sum',  # sum or mean
        'reward_sparsity_ts': 1,
        'reward_scale': 0.5,
        'reward_clipping': 0,
        'lambda_rtp': 0.01,

        # Network mods
        'gradient_clip_norm': 1,  # [0, 1, 5, 10],  # 0 is nothing
        'target_update_freq': 0.01,  # [50, 150, 500, 1e3, 1e4],  # consider n learning loops too
    }

    if selected_params['model'] == 1:
        # DQN-based
        architecture_params = {
            'network_size': [124, 124, 64]
        }
        selected_params = {**selected_params, **architecture_params}

    if selected_params['model'] == 2:
        # Dueling-DQN
        architecture_params = {
            'duel_shared_network_size': [512, 512],
            'duel_value_stream_size': [256, 256],
            'duel_advantage_stream_size': [256, 256]
        }
        selected_params = {**selected_params, **architecture_params}

    if selected_params['model'] == 3:
        # BDQ-based
        architecture_params = {
            'bdq_shared_network_size': [124, 124],
            'bdq_value_stream_size': [124, 64],
            'bdq_advantage_streams_size': [64, 64],

            'combine_reward': False,  # to keep zone reward contributions separate or not
            'td_target': 'mean',  # mean or max or empty ''

            'rescale_shared_grad_factor': 1 / (action_branches)
        }
        selected_params = {**selected_params, **architecture_params}

    if selected_params['rnn']:
        rnn_params = {
            # -- State Sequence --
            'sequence_ts_spacing': 3,
            'sequence_length': 4,  # input as list for variable ts spacing

            # -- BDQ Architecture --
            'lstm': True,
            'rnn_hidden_size': 128,
            'rnn_num_layers': 2,
        }
        selected_params = {**selected_params, **rnn_params}

    if selected_params['PER']:
        per_params = {
            'alpha_start': 1,
            'alpha_decay_factor': None,
            'betta_start': 0.6,
            'betta_decay_factor': 5e-6,
        }
        selected_params = {**selected_params, **per_params}

    Run = namedtuple('Run', selected_params.keys())
    selected_params = Run(*selected_params.values())

    # ----- For Hyperparameter Search -----
    hyperparameter_dict = {
        # -- Agent Params --
        'observation_ts_frequency': [5],  # * [5, 10, 15],
        'actuation_ts_frequency': [5],  # * [5, 10, 15],
        'learning_loops': [10],

        # --- Behavioral Policy ---
        'eps_start': [0.2],
        'eps_end': [0.01],
        'eps_decay': [1e-6],

        # --- Experience Replay ---
        'replay_capacity': [2048],
        'batch_size': [128],

        # DQN or BDQ
        'model': [2],  # 1=DQN, 2=Dueling DQN, 3=BDQ
        # PER
        'PER': [False],
        # RNN
        'rnn': [False],

        # -- BDQ --
        # Fixed
        'observation_dim': [18],
        'action_branches': [action_branches],  # n building zones
        'actuation_function': [5],  # ----------------------------------------------------------------------------------

        # TD Update
        'optimizer': ['Adagrad'],
        'learning_rate': [5e-1],
        'lr_scheduler': ['ReduceLROnPlateau'],
        'gamma': [0.9],

        # Reward
        'reward_aggregation': ['sum'],  # sum or mean
        'reward_sparsity_ts': [1],
        'reward_scale': [0.5],
        'reward_clipping': [0],
        'lambda_rtp': [0.3, 0.2, 0.1],

        # Network mods
        'gradient_clip_norm': [1],  # [0, 1, 5, 10],  # 0 is nothing
        'target_update_freq': [0.01],  # [50, 150, 500, 1e3, 1e4],  # consider n learning loops too
    }

    if 1 in hyperparameter_dict['model']:
        # DQN-based
        architecture_params = {
            'network_size': [[124, 124, 64]]
        }
        hyperparameter_dict = {**hyperparameter_dict, **architecture_params}

    if 2 in hyperparameter_dict['model']:
        # Dueling-DQN
        architecture_params = {
            'duel_shared_network_size': [[512, 512]],
            'duel_value_stream_size': [[256, 256]],
            'duel_advantage_stream_size': [[256, 256]]
        }
        hyperparameter_dict = {**hyperparameter_dict, **architecture_params}

    if 3 in hyperparameter_dict['model']:
        # BDQ-based
        architecture_params = {
            'bdq_shared_network_size': [[124, 124]],
            'bdq_value_stream_size': [[124, 64]],
            'bdq_advantage_streams_size': [[64, 64]],

            'combine_reward': [True],  # to keep zone reward contributions separate or not
            'td_target': ['mean'],  # mean or max or empty ''

            'rescale_shared_grad_factor': [1 / (action_branches)]
        }
        hyperparameter_dict = {**hyperparameter_dict, **architecture_params}

    if True in hyperparameter_dict['rnn']:
        rnn_params = {
            # -- State Sequence --
            'sequence_ts_spacing': [1, 6],
            'sequence_length': [6, 12],

            # -- BDQ Architecture --
            'lstm': [True],
            'rnn_hidden_size': [64, 128],
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
                    lstm=run.lstm,
                    lr_scheduler=run.lr_scheduler
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
                    lr_scheduler=run.lr_scheduler
                )

        # Dueling DQN
        elif run.model == 2:
            if run.rnn:
                self.dqn = DuelingDQN_RNN(
                    observation_dim=run.observation_dim,
                    rnn_hidden_size=run.rnn_hidden_size,
                    rnn_num_layers=run.rnn_num_layers,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.duel_shared_network_size,
                    value_stream_size=run.duel_value_stream_size,
                    advantage_stream_size=run.duel_advantage_stream_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    gradient_clip_norm=run.gradient_clip_norm,
                    lstm=run.lstm,
                    lr_scheduler=run.lr_scheduler
                )
            else:
                self.dqn = DuelingDQN(
                    observation_dim=run.observation_dim,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.duel_shared_network_size,
                    value_stream_size=run.duel_value_stream_size,
                    advantage_stream_size=run.duel_advantage_stream_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    gradient_clip_norm=run.gradient_clip_norm,
                    lr_scheduler=run.lr_scheduler
                )
        # BDQ
        elif run.model == 3:
            if run.rnn:
                self.dqn = BranchingDQN_RNN(
                    observation_dim=run.observation_dim,
                    rnn_hidden_size=run.rnn_hidden_size,
                    rnn_num_layers=run.rnn_num_layers,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.bdq_shared_network_size,
                    value_stream_size=run.bdq_value_stream_size,
                    advantage_streams_size=run.bdq_advantage_streams_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    td_target=run.td_target,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor,
                    lstm=run.lstm,
                    lr_scheduler=run.lr_scheduler
                )
            else:
                self.dqn = BranchingDQN(
                    observation_dim=run.observation_dim,
                    action_branches=run.action_branches,
                    action_dim=Agent.actuation_function_dim(actuation_function_id=run.actuation_function),
                    shared_network_size=run.bdq_shared_network_size,
                    value_stream_size=run.bdq_value_stream_size,
                    advantage_streams_size=run.bdq_advantage_streams_size,
                    target_update_freq=run.target_update_freq,
                    learning_rate=run.learning_rate,
                    optimizer=run.optimizer,
                    gamma=run.gamma,
                    td_target=run.td_target,
                    gradient_clip_norm=run.gradient_clip_norm,
                    rescale_shared_grad_factor=run.rescale_shared_grad_factor,
                    lr_scheduler=run.lr_scheduler
                )

        return self.dqn
