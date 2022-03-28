import copy
import random
from collections import namedtuple
from itertools import product

import torch.utils.tensorboard as tb

from emspy import BcaEnv, MdpManager
from bca import BranchingDQN, ReplayMemory, EpsilonGreedyStrategy, Agent_TB


class RunManager:
    """This class helps manage all hparams and sampling, as well as creating all objects dependent on these hparams."""
    # -- Agent Params --
    # Misc. Params
    action_branches = 4

    selected_params = {
        # -- Agent Params --
        'interaction_ts_frequency': [5],  # * [5, 10, 15],
        'learning_loops': [10],

        # --- Behavioral Policy ---
        'eps_start': [0.15],
        'eps_end': [0.05],
        'eps_decay': [1e-5],

        # --- Experience Replay ---
        'replay_capacity': [500],
        'batch_size': [32],

        # -- BDQ --
        # Fixed
        'observation_dim': [36],
        'action_branches': [action_branches],  # n building zones
        'action_dim': [6],

        # Architecture
        'shared_network_size_l1': [96],
        'shared_network_size_l2': [64],
        'value_stream_size': [48],
        'advantage_streams_size': [48],

        # TD Update
        'learning_rate': [1e-4],
        'gamma': [0.7],

        # Network mods
        'td_target': [1],  # (0) mean or (1) max
        'gradient_clip_norm': [1],  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': [1 / (1 + action_branches)],
        'target_update_freq': [1e3]  # [50, 150, 500, 1e3, 1e4],
    }

    agent_params = {
        'interaction_ts_frequency': [5],  # * [5, 10, 15],
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
        'observation_dim': [36],
        'action_branches': [action_branches],  # n building zones
        'action_dim': [6],
    }

    bdq_params = {
        # --- BDQ ---
        # Architecture
        'shared_network_size_l1': [96],
        'shared_network_size_l2': [64],
        'value_stream_size': [48],
        'advantage_streams_size': [48],

        # TD Update
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'gamma': [0.3, 0.5, 0.7],

        # Network mods
        'td_target': [1],  # (0) mean or (1) max
        'gradient_clip_norm': [1],  # [0, 1, 5, 10],  # 0 is nothing
        'rescale_shared_grad_factor': [1 / (1 + action_branches)],
        'target_update_freq': [500.0, 1e3, 2e3]  # [50, 150, 500, 1e3, 1e4],
    }

    hyperparameter_dict = {**agent_params, **bdq_fixed_params, **bdq_params}

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

    def get_runs(self, params: dict):
        """Get all permutations of hyperparameters passed."""
        Run = namedtuple('Run', params.keys())

        runs = []
        for config in product(*params.values()):
            run_config = Run(*config)
            if run_config.batch_size > run_config.replay_capacity:
                # Incompatible hyperparam configuration
                pass
            else:
                runs.append(run_config)

        return runs

    def shuffle_runs(self):
        """Returns list of shuffled runs"""

        runs = copy.copy(self.runs)
        random.shuffle(runs)

        return runs

    def create_agent(self, run, mdp: MdpManager, sim: BcaEnv, summary_writer: tb.SummaryWriter):
        """Creates and returns new RL Agent from defined parameters."""

        self.agent = Agent_TB(
            emspy_sim=sim,
            mdp=mdp,
            dqn_model=self.dqn,
            policy=self.policy,
            replay_memory=self.experience_replay,
            interaction_frequency=run.interaction_ts_frequency,
            learning_loop=run.learning_loops,
            summary_writer=summary_writer
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

    def create_exp_replay(self, run: namedtuple):
        """Creates and returns new Experience Replay from defined parameters."""

        self.experience_replay = ReplayMemory(
            capacity=run.replay_capacity,
            batch_size=run.batch_size
        )

        return self.experience_replay

    def create_bdq(self, run: namedtuple):
        """Creates and returns new BDQ model from defined parameters."""

        self.dqn = BranchingDQN(
            observation_dim=run.observation_dim,
            action_branches=run.action_branches,
            action_dim=run.action_dim,
            shared_network_size=[run.shared_network_size_l1, run.shared_network_size_l2],
            value_stream_size=[run.value_stream_size],
            advantage_streams_size=[run.advantage_streams_size],
            target_update_freq=run.target_update_freq,
            learning_rate=run.learning_rate,
            gamma=run.gamma,
            td_target=run.td_target,
            gradient_clip_norm=run.gradient_clip_norm,
            rescale_shared_grad_factor=run.rescale_shared_grad_factor
        )

        return self.dqn
