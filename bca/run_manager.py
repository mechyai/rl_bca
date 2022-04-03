import copy
import random
from collections import namedtuple
from itertools import product


from emspy import BcaEnv, MdpManager
from bca import BranchingDQN, BranchingDQN_RNN
from bca import ReplayMemory, SequenceReplayMemory, EpsilonGreedyStrategy, Agent_TB
from bca import TensorboardManager


class RunManager:
    """This class helps manage all hparams and sampling, as well as creating all objects dependent on these hparams."""
    # -- Agent Params --
    # Misc. Params
    action_branches = 4

    selected_params = {
        # -- Agent Params --
        'observation_ts_frequency': 5,  # * [5, 10, 15],
        'actuation_ts_frequency': 5,  # * [5, 10, 15],
        'learning_loops': 10,

        # --- Behavioral Policy ---
        'eps_start': 0.1,
        'eps_end': 0.025,
        'eps_decay': 1e-3,

        # --- Experience Replay ---
        'replay_capacity': 1000,
        'batch_size': 64,

        # -- BDQ --
        # Fixed
        'observation_dim': 60,
        'action_branches': action_branches,  # n building zones
        'actuation_function': 1,
        'actuation_dim': 3,

        # Architecture
        'shared_network_size_l1': 96,
        'shared_network_size_l2': 96,
        'value_stream_size': 48,
        'advantage_streams_size': 48,

        # TD Update
        'reward_aggregation': 'mean',  # sum or mean
        'optimizer': 'Adagrad',
        'learning_rate': 1e-3,
        'gamma': 0.7,

        # Network mods
        'td_target': 'mean',  # (0) mean or (1) max
        'gradient_clip_norm': 1,  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': 1 / (action_branches),
        'target_update_freq': 1e3,  # [50, 150, 500, 1e3, 1e4],

        # RNN
        # -- Agent / Model --
        'rnn': False,

        # -- Replay Memory --
        'sequence_ts_spacing': 3,
        'sequence_length': 5,

        # -- BDQ Architecture --
        'rnn_hidden_size': 64,
        'rnn_num_layers': 2,
    }
    Run = namedtuple('Run', selected_params.keys())
    selected_params = Run(*selected_params.values())

    rnn_params = {
        # -- Agent / Model --

        'rnn': [True, False],

        # -- Replay Memory --
        'sequence_ts_spacing': [1, 3],
        'sequence_length': [5, 10],

        # -- BDQ Architecture --
        'rnn_hidden_size': [64],
        'rnn_num_layers': [1, 2],
    }

    agent_params = {
        'observation_ts_frequency': [5],  # * [5, 10, 15],
        'actuation_ts_frequency': [5],  # * [5, 10, 15],
        'learning_loops': [10],

        # --- Behavioral Policy ---
        'eps_start': [0.15],
        'eps_end': [0.05],
        'eps_decay': [1e-5],

        # --- Experience Replay ---
        'replay_capacity': [500, 1000, 5000],
        'batch_size': [32, 64, 128],
    }

    bdq_fixed_params = {
        'observation_dim': [64],
        'action_branches': [action_branches],  # n building zones
        'actuation_function': [1],
        'actuation_dim': [3],
    }

    bdq_params = {
        # --- BDQ ---
        # Architecture
        'shared_network_size_l1': [96],
        'shared_network_size_l2': [],
        'value_stream_size': [48],
        'advantage_streams_size': [48],

        # TD Update
        'reward_aggregation': ['mean'],  # sum or mean
        'optimizer': ['Adam'],
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'gamma': [0.3, 0.5, 0.7],

        # Network mods
        'td_target': ['max'],  # mean or max
        'gradient_clip_norm': [1],  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': [1 / (1 + action_branches)],
        'target_update_freq': [500.0, 1e3, 2e3]  # [50, 150, 500, 1e3, 1e4],
    }

    # hyperparameter_dict = {**agent_params, **bdq_fixed_params, **bdq_params}
    hyperparameter_dict = {**agent_params, **bdq_fixed_params, **bdq_params, **rnn_params}

    # The hyperparameters that vary throughout a study
    hyperparameter_study = {}
    for key, value in hyperparameter_dict.items():
        if len(value) > 1:
            hyperparameter_study[key] = value

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

        actuation_functions = params['actuation_function']
        actuation_dimensions = params['actuation_dim']

        runs = []
        for config in product(*params.values()):
            run_config = Run(*config)
            # Handle incompatible hyperparam configuration
            if run_config.batch_size > run_config.replay_capacity:
                pass
            else:
                # TODO fix this, so bad
                # Link actuation functions with their intended dimensions, no permutation here
                func_index = actuation_functions.index(run_config.actuation_function)
                # Now could end up with duplicate permutations
                run_config = run_config._replace(actuation_dim=actuation_dimensions[func_index])

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

    def create_agent(self, run, mdp: MdpManager, sim: BcaEnv, tensorboard_manager: TensorboardManager):
        """Creates and returns new RL Agent from defined parameters."""

        self.agent = Agent_TB(
            emspy_sim=sim,
            mdp=mdp,
            dqn_model=self.dqn,
            policy=self.policy,
            replay_memory=self.experience_replay,
            rnn=run.rnn,
            observation_frequency=run.observation_ts_frequency,
            actuation_frequency=run.actuation_ts_frequency,
            actuation_dimension=run.actuation_dim,
            reward_aggregation=run.reward_aggregation,
            learning_loop=run.learning_loops,
            tensorboard_manager=tensorboard_manager
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
            self.experience_replay = SequenceReplayMemory(
                capacity=run.replay_capacity,
                batch_size=run.batch_size,
                sequence_length=run.sequence_length,
                sequence_ts_spacing=run.sequence_ts_spacing
            )
        else:
            self.experience_replay = ReplayMemory(
                capacity=run.replay_capacity,
                batch_size=run.batch_size
            )

        return self.experience_replay

    def create_bdq(self, run: namedtuple):
        """Creates and returns new BDQ model from defined parameters."""

        if run.rnn:
            self.dqn = BranchingDQN_RNN(
                observation_dim=run.observation_dim,
                rnn_hidden_size=run.rnn_hidden_size,
                rnn_num_layers=run.rnn_num_layers,
                action_branches=run.action_branches,
                action_dim=run.run.actuation_dim,
                shared_network_size=[run.shared_network_size_l1, run.shared_network_size_l2],
                value_stream_size=[run.value_stream_size],
                advantage_streams_size=[run.advantage_streams_size],
                target_update_freq=run.target_update_freq,
                learning_rate=run.learning_rate,
                optimizer=run.optimizer,
                gamma=run.gamma,
                td_target=run.td_target,
                gradient_clip_norm=run.gradient_clip_norm,
                rescale_shared_grad_factor=run.rescale_shared_grad_factor
            )

        else:
            self.dqn = BranchingDQN(
                observation_dim=run.observation_dim,
                action_branches=run.action_branches,
                action_dim=run.actuation_dim,
                shared_network_size=[run.shared_network_size_l1, run.shared_network_size_l2],
                value_stream_size=[run.value_stream_size],
                advantage_streams_size=[run.advantage_streams_size],
                target_update_freq=run.target_update_freq,
                learning_rate=run.learning_rate,
                optimizer=run.optimizer,
                gamma=run.gamma,
                td_target=run.td_target,
                gradient_clip_norm=run.gradient_clip_norm,
                rescale_shared_grad_factor=run.rescale_shared_grad_factor
            )

        return self.dqn
