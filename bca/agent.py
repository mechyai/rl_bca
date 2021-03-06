import math
import time
import itertools
import random
from typing import Union
from datetime import datetime as dt
from datetime import timedelta
import calendar

import numpy as np
import torch

from emspy import BcaEnv, MdpManager

from bca import BranchingDQN, EpsilonGreedyStrategy, ReplayMemory, PrioritizedReplayMemory
from bca import BranchingDQN_RNN, SequenceReplayMemory, PrioritizedSequenceReplayMemory
from bca import MDP

# -- Normalization Params --
hvac_electricity_energy = {
    # Zn0
    # 'zn0_heating_electricity_max': 688000,  # [J]
    'zn0_cooling_electricity_max': MDP.tc_meters['zn0_cooling_electricity'][3],  # [J]
    'zn0_fan_electricity_max': MDP.tc_meters['zn3_fan_electricity'][3],  # [J]
    # Zn1
    # 'zn1_heating_electricity_max': 22000,  # [J]
    'zn1_cooling_electricity_max': MDP.tc_meters['zn1_cooling_electricity'][3],  # [J]
    'zn1_fan_electricity_max': MDP.tc_meters['zn3_fan_electricity'][3],  # [J]
    # Zn2
    # 'zn2_heating_electricity_max': 128000,  # [J]
    'zn2_cooling_electricity_max': MDP.tc_meters['zn2_cooling_electricity'][3],  # [J]
    'zn2_fan_electricity_max': MDP.tc_meters['zn3_fan_electricity'][3],  # [J]
    # Zn3
    # 'zn3_heating_electricity_max': 149000,  # [J]
    'zn3_cooling_electricity_max': MDP.tc_meters['zn3_cooling_electricity'][3],  # [J]
    'zn3_fan_electricity_max': MDP.tc_meters['zn3_fan_electricity'][3],  # [J]
    # Zn4
    # 'zn4_heating_electricity_max': None,
    'zn4_cooling_electricity_max': None,
    'zn4_fan_electricity_max': None
}


class Agent:
    @staticmethod
    def actuation_function_dim(actuation_function_id):
        """Returns the number of action dimensions linked to the particular actuation function"""
        action_dim_directory = {
            1: 3,  # act_heat_cool_off_1
            2: 6,  # act_strict_setpoints_2
            3: 3,  # act_step_strict_setpoints_3
            4: 6,  # act_default_adjustments_4
            5: 9,  # act_cool_only_5
            6: 7,  # act_cool_only_default_adjustment_6
            7: 2,  # act_cool_only_on_off_7
            8: 3,  # act_cool_only_on_off_stay_8
        }

        return action_dim_directory[actuation_function_id]

    def __init__(self,
                 emspy_sim: BcaEnv,
                 mdp: MdpManager,
                 bem_model,  # TODO fix imports
                 dqn_model: Union[BranchingDQN, BranchingDQN_RNN],
                 policy: EpsilonGreedyStrategy,
                 replay_memory: Union[ReplayMemory, PrioritizedReplayMemory, SequenceReplayMemory],
                 run_parameters,
                 observation_frequency: int,
                 actuation_frequency: int,
                 actuation_dimension: int,
                 rnn: bool = False,
                 reward_aggregation: str = 'sum',
                 learning_loop: int = 1,
                 tensorboard_manager=None,  # TODO fix imports
                 current_step: int = 0,
                 continued_parameters: dict = None,
                 print_values: bool = False
                 ):

        # -- SIMULATION STATES --
        self.sim = emspy_sim
        self.mdp = mdp
        # EMS MdpElement objects
        self.vars_mdp_elements = mdp.ems_type_dict['var']
        self.meters_mdp_elements = mdp.ems_type_dict['meter']
        self.weather_mdp_elements = mdp.ems_type_dict['weather']
        # ToC EMS names
        self.var_names = list(self.mdp.tc_var.keys())
        self.meter_names = list(self.mdp.tc_meter.keys())
        self.weather_names = list(self.mdp.tc_weather.keys())
        # EMS values
        self.time = None
        self.var_vals = None
        self.meter_vals = None
        self.weather_vals = None
        # EMS encoded values
        self.var_encoded_vals = None
        self.meter_encoded_vals = None
        self.weather_encoded_vals = None

        # -- STATE SPACE --
        self.state_var_names = {}
        self.termination = 0
        self.state_normalized = None
        self.next_state_normalized = None

        # -- ACTION SPACE --
        self.actuation_dim = actuation_dimension
        self.action = None
        self.actuation_dict = {}
        self.epsilon = policy.start
        self.fixed_epsilon = None  # optional fixed exploration rate
        self.greedy_epsilon = policy

        # -- ACTION ENCODING --
        self.temp_deadband = 5  # distance between heating and cooling setpoints
        self.temp_buffer = 1  # new setpoint distance from current temps
        self.current_setpoint_windows = [3, 3, 3, 3]

        # -- REWARD --
        self.reward_aggregation = reward_aggregation
        self.reward_dict = None
        self.reward = 0
        self.reward_sum = 0
        self.reward_component_sum = [0, 0, 0]
        self.reward_zone_sum = [0] * dqn_model.action_branches

        # -- CONTROL GOALS --
        self.indoor_temp_ideal_range = np.array([21.1, 23.89])  # occupied hours, based on OS model
        self.indoor_temp_unoccupied_range = np.array(
            [15.6 - 0.5, 29.4 + 0.5])  # mimic night cycle manager, + 1/2 temp tolerance
        self.indoor_temp_limits = np.array([15, 30])  # ??? needed?
        self.setpoint_deadband = 2.5  # distance between cooling and heating sp

        # -- TIMING --
        self.weekend_days = ['Friday', 'Saturday', 'Sunday', 'Monday']
        self.week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        self.n_ts = 0
        self.current_step = current_step
        self.friday_date = 0
        self.simulation_start_time = None

        # -- INTERACTION FREQUENCIES --
        self.observation_frequency = observation_frequency
        self.actuation_frequency = actuation_frequency

        # -- REPLAY MEMORY --
        self.memory = replay_memory
        # PER
        self.alpha_start = None
        self.betta_start = None

        # -- BDQ --
        self.dqn_model = dqn_model
        self.rnn = rnn

        # -- PERFORMANCE RESULTS --
        self.comfort_dissatisfaction = 0
        self.hvac_rtp_costs = 0
        self.comfort_dissatisfaction_total = 0
        self.hvac_rtp_costs_total = 0

        # -- RESULTS TRACKING --
        # Comfort
        self.cold_temps_histogram_data = np.array([])
        self.warm_temps_histogram_data = np.array([])
        # RTP $
        self.rtp_histogram_data = []
        # Wind Energy
        self.wind_energy_hvac_data = []
        self.total_energy_hvac_data = []
        # TensorBoard
        self.TB = tensorboard_manager

        # -- LEARNING --
        self.learning = True
        self.learning_steps = 0
        self.learning_loop = learning_loop
        self.loss = 0
        self.loss_total = 0

        # -- Misc. --
        self.run = run_parameters
        self.bem = bem_model
        self.rnn_start = False
        self._once = True
        self._print = print_values
        self._checked_action_dims = False

        # -- Parameter Tracking --
        self.continued_parameters = continued_parameters
        self._set_continued_params()

    # ----------------------------------------------------- STATE -----------------------------------------------------

    def observe(self, learn=True):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.next_state_normalized = self._get_encoded_state()

        if self._print:
            print(f'\n\n{str(self.time)}\n')
            # print(f'\n\n\tVars: {vars}\n\tMeters: {meters}\n\tWeather: {weather}')
            print(f'\n\t{self.next_state_normalized}\n')

        # -- TERMINAL STATE --
        self.termination = self._is_terminal()

        # -- REWARD --
        self.reward_dict = self._reward4()
        # Aggregate reward across zones
        self.reward = self._get_total_reward(self.reward_aggregation) * self.run.reward_scale
        # Reward clipping
        if self.run.reward_clipping != 0:
            self.reward = np.fmax(self.reward, self.run.reward_clipping)

        # Get total reward per component
        reward_component_sums = np.array(list(self.reward_dict.values())).sum(axis=0) * self.run.reward_scale  # sum reward per component
        self.reward_component_sum = np.array(list(zip(self.reward_component_sum, reward_component_sums))).sum(axis=1)
        # Get total reward per zone
        reward_zone_sums = np.array(list(self.reward_dict.values())).sum(axis=1) * self.run.reward_scale  # sum reward per zone
        self.reward_zone_sum = np.array(list(zip(self.reward_zone_sum, reward_zone_sums))).sum(axis=1)

        # -- STORE INTERACTIONS --
        # Must store regardless for RNN to work
        if (learn or self.rnn) and self.action is not None:  # after first action, enough data available
            # <S, A, S', R, t> - push experience to Replay Memory
            self.memory.push(
                self.state_normalized,
                self.action,
                self.next_state_normalized,
                self.reward,
                self.termination
            )

        # -- LEARN BATCH --
        if learn:
            # Enough memory for batch
            if self.memory.can_provide_sample():
                # Learning Loop
                for i in range(self.learning_loop):
                    self.learning_steps += 1
                    # If PER
                    if isinstance(self.memory, PrioritizedReplayMemory) \
                            or isinstance(self.memory, PrioritizedSequenceReplayMemory):
                        # Get prioritized batch
                        batch, sample_indices = self.memory.sample()
                        # Learn from prioritized batch
                        weights = self.memory.get_gradient_weights(sample_indices)
                        self.loss, loss_each = self.dqn_model.update_policy(batch, gradient_weights=weights)
                        # Update replay priorities
                        self.memory.update_td_losses(sample_indices, loss_each)
                    else:
                        # Get random batch
                        batch = self.memory.sample()
                        self.loss, loss_each = self.dqn_model.update_policy(batch)  # batch learning

                    # Update data
                    self.loss_total += self.loss

                # -- ANNEAL LEARNING VARS --
                if isinstance(self.memory, PrioritizedReplayMemory) \
                        or isinstance(self.memory, PrioritizedSequenceReplayMemory):
                    self.decay_alpha_betta()

            # -- ANNEAL INTERACTION VARS --
            # Adaptive Learning Rate
            # self.bdq.update_learning_rate()

        # -- PERFORMANCE RESULTS --
        # Record results
        self.comfort_dissatisfaction = self._get_comfort_results()
        self.hvac_rtp_costs = self._get_rtp_hvac_cost_results()
        # # Update Results Sums
        self.comfort_dissatisfaction_total += self.comfort_dissatisfaction
        self.hvac_rtp_costs_total += self.hvac_rtp_costs
        self.reward_sum += self.reward

        # Don't record results until SequenceMemory (RNN) has enough results to inference
        if not self.rnn_start and self.run.rnn and learn:
            # Reset performance results
            self.comfort_dissatisfaction = 0
            self.hvac_rtp_costs = 0
            self.comfort_dissatisfaction = 0
            self.hvac_rtp_costs_total = 0
            # Reset reward
            self.reward = np.zeros_like(self.reward)
            self.reward_sum = np.zeros_like(self.reward_sum)
            self.reward_component_sum = [0, 0, 0]
            self.reward_zone_sum = [0] * self.dqn_model.action_branches

        # -- UPDATE DATA --
        self.state_normalized = self.next_state_normalized
        self.current_step += 1

        # -- TensorBoard --
        if not self.run.rnn or self.rnn_start or not learn:
            # Collect results right away for non-RNN models, or wait until rnn_start
            self.TB.record_timestep_results(self)

        # -- REPORTING --
        if self._print:
            print(f'\n\t*Reward: {round(self.reward.sum(), 2)}, Cumulative: {round(self.reward_sum.sum(), 2)}')
            pass

        # -- TRACK REWARD --
        # return sum([self.reward] if not hasattr(self.reward, '__iter__') else self.reward)

    def _get_encoded_state(self):
        """Gets and processes state input from simulation at every timestep. Returns the current encoded state."""

        self.time = self.sim.get_ems_data(['t_datetimes'])
        self.var_vals = self.mdp.update_ems_value_from_dict(self.sim.get_ems_data(self.var_names, return_dict=True))
        self.meter_vals = self.mdp.update_ems_value_from_dict(self.sim.get_ems_data(self.meter_names, return_dict=True))
        self.weather_vals = self.mdp.update_ems_value_from_dict(self.sim.get_ems_data(self.weather_names, return_dict=True))

        # -- MODIFY STATE --
        meter_names = [meter for meter in self.meter_names if 'fan' not in meter]  # remove unwanted fan meters

        # -- GET ENCODING --
        self.var_encoded_vals = self.mdp.get_ems_encoded_values(self.var_names)
        self.meter_encoded_vals = self.mdp.get_ems_encoded_values(meter_names)
        self.weather_encoded_vals = self.mdp.get_ems_encoded_values(self.weather_names)

        # -- REMOVE Uncontrolled Zones --
        for zone_i_exclude in range(self.dqn_model.action_branches, 6):  # 5 total zones
            # Manage meters
            for key in dict(self.meter_encoded_vals).keys():
                if f'zn{zone_i_exclude}' in key:
                    del self.meter_encoded_vals[key]
            # Manage vars
            for key in dict(self.var_encoded_vals).keys():
                if f'zn{zone_i_exclude}' in key:
                    del self.var_encoded_vals[key]

        # -- Combine Heating & Cooling Electricity --
        # for meter_name in self.meter_encoded_vals.copy():
        #     # combine heating and cooling into 1 val [-1, 0]:cooling + [0, 1]:heating, then remove individuals
        #     if 'heating' in meter_name:
        #         zone_n = meter_name.split('_')[0]
        #         heating_val = self.meter_encoded_vals.pop(meter_name)
        #         cooling_val = self.meter_encoded_vals.pop(zone_n + '_cooling_electricity')
        #         self.meter_encoded_vals[zone_n + '_hvac_electricity'] = heating_val + cooling_val

        # ----------------------------------- Temperature Bounds Warning -----------------------------------
        occupancy_schedule = self.mdp.get_mdp_element('hvac_operation_sched').value
        temperature_warnings_list = []
        for zone_i in range(self.dqn_model.action_branches):
            zone_temp = self.mdp.get_mdp_element(f'zn{zone_i}_temp').value
            # Occupied hours
            if occupancy_schedule == 1:
                if zone_temp >= self.indoor_temp_ideal_range[1]:
                    warning_values = 1
                elif zone_temp <= self.indoor_temp_ideal_range[0]:
                    warning_values = -1
                else:
                    warning_values = 0
            # Unoccupied hours
            else:
                if zone_temp >= self.indoor_temp_unoccupied_range[1]:
                    warning_values = 1
                elif zone_temp <= self.indoor_temp_unoccupied_range[0]:
                    warning_values = -1
                else:
                    warning_values = 0
            temperature_warnings_list.append(warning_values)

        # ----------------------------------- RTP High-Price Signal -----------------------------------
        rtp = self.var_vals['rtp']
        # Add extra RTP pricing state signal
        if rtp > 50:
            rtp_alert = [1]
        elif rtp < 15:
            rtp_alert = [-1]
        else:
            rtp_alert = [0]

        # ----------------------------------- Weather Forecast -----------------------------------
        weather_forecast_list = []
        hours_ahead = 1
        for hour in range(1, hours_ahead + 1, 1):
            current_hour = self.time.hour
            forecast_day = 'today' if current_hour + hour < 24 else 'tomorrow'
            forecast_hour = (current_hour + hour) % 24  # E+ clock is 0-23 hrs

            weather_forecast_list.append(
                MDP.normalize_min_max_saturate(
                    self.sim.get_weather_forecast(['oa_db'], forecast_day, forecast_hour, zone_ts=1),
                    MDP.outdoor_temp_min,
                    MDP.outdoor_temp_max)
            )
            weather_forecast_list.append(
                MDP.digitize_bool(
                    self.sim.get_weather_forecast(['sun_up'], forecast_day, forecast_hour, zone_ts=1))
            )

        # ----------------------------------- Timing -----------------------------------
        year = self.time.year
        month = self.time.month
        days_of_month = calendar.monthrange(self.time.year, month)[1]
        day = self.time.day
        hour = self.time.hour
        minute = self.time.minute

        time_list = [month / 12, day / days_of_month, hour / 24, minute / 60]

        # ----------------------------------- Building Schedule Progress -----------------------------------
        hour_start = 6
        hour_end = 19
        weekend = False
        day_name = self.time.strftime("%A")

        # Handle Weekend Progress
        if day_name in self.weekend_days:
            # Get weekend end and start, handle potential change in month/year
            friday = self.time - timedelta(days=self.weekend_days.index(day_name))  # subtract to Friday
            monday = self.time + timedelta(days=(3 - self.weekend_days.index(day_name)))  # add to Monday
            weekend_start = dt(friday.year, friday.month, friday.day, hour_end, 0)  # end of workday Friday
            weekend_end = dt(monday.year, monday.month, monday.day, hour_start, 0)  # start of workday Monday

            # Get progress into weekend
            if weekend_start <= self.time <= weekend_end:
                weekend = True
                building_hours_progress = (self.time - weekend_start).total_seconds() \
                                          / (weekend_end - weekend_start).total_seconds()
                week_state_hot_encoding = [0, 0, 1]  # Weekend

        # Handle Weekday Progress
        if day_name in self.week_days and not weekend:
            workday_start = dt(year, month, day, hour_start, 0)  # start of workday today
            workday_end = dt(year, month, day, hour_end, 0)  # end of workday today
            # During workday
            if workday_start <= self.time <= workday_end:
                building_hours_progress = (self.time - workday_start).total_seconds() \
                                          / (workday_end - workday_start).total_seconds()
                week_state_hot_encoding = [1, 0, 0]  # Work
            # Before workday
            elif self.time < workday_start:
                since_workday_end = (24 - hour_end + hour) * 3600 + minute * 60  # seconds since previous day end
                building_hours_progress = since_workday_end / ((24 - hour_end + hour_start) * 3600)
                week_state_hot_encoding = [0, 1, 0]  # Off Work
            # After workday
            elif self.time > workday_end:
                building_hours_progress = (self.time - workday_end).total_seconds() \
                                          / ((24 - hour_end + hour_start) * 3600)
                week_state_hot_encoding = [0, 1, 0]  # Off Work
        elif not weekend:
            # Catch any errors
            building_hours_progress = None
            week_state_hot_encoding = None

        building_hours_progress = [*MDP.sin_cos_normalization(building_hours_progress)]

        # ----------------------------------- State History -----------------------------------
        # Prior temps
        prior_timesteps = 1
        prior_timestep_spacing = 1
        prior_timesteps_list = list(range(1, prior_timesteps + 1, prior_timestep_spacing))
        indoor_temp_vars = [f'zn{zn_num}_temp' for zn_num in range(self.dqn_model.action_branches)]
        outdoor_var = ['oa_db']

        prior_indoor_temp_data = self.sim.get_ems_data(indoor_temp_vars, prior_timesteps_list, return_dict=False)
        prior_outdoor_data = self.sim.get_ems_data(outdoor_var, prior_timesteps_list, return_dict=False)

        prior_data = []
        # Indoor Temps
        for zone_data_times in (
                [prior_indoor_temp_data] if not hasattr(prior_indoor_temp_data,
                                                        '__iter__') else prior_indoor_temp_data):
            for zone_temps in [zone_data_times]:
                prior_data.append(MDP.normalize_min_max_saturate(zone_temps, MDP.indoor_temp_min, MDP.indoor_temp_max))
        # Outdoor conditions
        for outdoor_times in [prior_outdoor_data]:
            prior_data.append(MDP.normalize_min_max_saturate(outdoor_times, MDP.outdoor_temp_min, MDP.outdoor_temp_max))

        # -- DO ONCE --
        if self._once:
            self._once = False
            self.state_var_names = self.var_names + self.weather_names + meter_names
            self.simulation_start_time = self.time

        # -- ENCODED STATE --
        return np.array(
            temperature_warnings_list +
            rtp_alert +
            # time_list +
            # weather_forecast_list +
            building_hours_progress +
            week_state_hot_encoding +
            # prior_data +
            list(self.var_encoded_vals.values()) +
            list(self.weather_encoded_vals.values()) +
            list(self.meter_encoded_vals.values()),
            dtype=float)

    # ------------------------------------------------- Misc... -------------------------------------------------

    def _set_continued_params(self):
        """Write saved parameters from previous episode."""
        if self.continued_parameters is not None:
            if self.run.PER:
                if 'alpha_start' in self.continued_parameters:
                    self.alpha_start = self.continued_parameters['alpha_start']
                if 'betta_start' in self.continued_parameters:
                    self.betta_start = self.continued_parameters['betta_start']
            if 'epsilon_start' in self.continued_parameters:
                self.greedy_epsilon.start = self.continued_parameters['epsilon_start']
        else:
            self.alpha_start = 1
            self.betta_start = 1

    def save_continued_params(self):
        """Save parameters from previous episode."""
        if self.run.PER:
            if 'alpha_start' in self.continued_parameters:
                self.continued_parameters['alpha_start'] = self.memory.alpha
            if 'betta_start' in self.continued_parameters:
                self.continued_parameters['betta_start'] = self.memory.betta
        if 'epsilon_start' in self.continued_parameters:
            self.continued_parameters['epsilon_start'] = self.epsilon

        return self.continued_parameters

    def _is_terminal(self):
        """Determines whether the current state is a terminal state or not. Dictates TD update values."""
        if self.time.day > self.bem.end_day:  # end of sim, goes to next day 0-hour
            # Terminal state
            return 1
        return 0

    def decay_alpha_betta(self):
        """Anneal variables of prioritization (alpha) and gradient weight adjustments (betta) with annealing."""

        if self.run.PER:
            alpha_start = self.alpha_start
            # alpha_growth_factor = self.run.alpha_decay_factor
            betta_start = self.betta_start
            betta_decay_factor = self.run.betta_decay_factor

            # self.memory.alpha = alpha_start * math.exp(-alpha_decay_factor * self.learning_steps)  # 1 --> 0
            self.memory.alpha = alpha_start
            self.memory.betta = min(1 - (1 - betta_start) * math.exp(-betta_decay_factor * self.learning_steps),
                                    1)  # 0 --> 1

    # ------------------------------------------------- ACTUATION -------------------------------------------------

    def _get_aux_actuation(self):
        """
        Used to manage auxiliary actuation (likely schedule writing) in one place.
        """
        reward_component_instance = np.array(list(self.reward_dict.values())).sum(axis=0)

        # Data Tracking
        return {
            # -- Rewards --
            'reward': self.reward.sum(),
            'reward_cumulative': self.reward_sum.sum(),
            # Reward Components
            'reward_comfort': reward_component_instance[0],
            'reward_cumulative_comfort': self.reward_component_sum[0],
            'reward_rtp': reward_component_instance[1],
            'reward_cumulative_rtp': self.reward_component_sum[1],
            # 'reward_wind': reward_component_instance[2],
            # 'reward_cumulative_wind': self.reward_component_sum[2],

            # -- Results Metric --
            # Comfort
            'comfort': self.comfort_dissatisfaction,
            'comfort_cumulative': self.comfort_dissatisfaction_total,
            # RTP
            'rtp_tracker': self.hvac_rtp_costs,
            'rtp_cumulative': self.hvac_rtp_costs_total,
            # Wind
            # 'wind_hvac_use': self.wind_energy_hvac_data[-1],
            # 'total_hvac_use': self.total_energy_hvac_data[-1],
            # -- Learning --
            'loss': self.loss,
            'loss_cumulative': self.loss_total
        }

    def _exploit_action(self):
        """Function to handle nuances of exploiting actions. Handles special case for RNN BDQ."""

        if self.rnn:
            if self.memory.current_interaction_count > self.memory.sequence_index_span:
                # Need to have full sequence
                self.rnn_start = True
                self.action = self.dqn_model.get_greedy_action(self.memory.get_single_sequence())

                return 'Exploit'
            else:
                # Temporary explore until full sequence available
                if self.run.model == 3:
                    # BDQ-based
                    self.action = np.random.randint(0, self.dqn_model.action_dim, self.dqn_model.action_branches)
                else:
                    # DQN-based
                    self.action = random.randint(0, self.dqn_model.action_dim ** self.dqn_model.action_branches - 1)

                return 'Explore'
        else:
            self.action = self.dqn_model.get_greedy_action(torch.Tensor(self.state_normalized).unsqueeze(1))

            return 'Exploit'

    def _explore_exploit_process(self, exploit: bool):
        """Helper function to handle explore/exploit decision of actions."""

        self.epsilon = self.greedy_epsilon.get_exploration_rate(self.current_step, self.fixed_epsilon)
        if not exploit and np.random.random() < self.epsilon:
            # Explore
            if self.run.model == 3:
                # BDQ-based
                self.action = np.random.randint(0, self.dqn_model.action_dim, self.dqn_model.action_branches)
            else:
                # DQN-based
                self.action = random.randint(0, self.dqn_model.action_dim ** self.dqn_model.action_branches - 1)

            action_type = 'Explore'
        else:
            # Exploit (handle RNN)
            action_type = self._exploit_action()

        if self._print:
            print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

    def _action_dimension_check(self, this_actuation_functions_dims=0):
        """Used to verify that action dimensions aligns with BDQ architecture. Raises error and exits if not."""

        if not self._checked_action_dims:
            if self.actuation_dim != this_actuation_functions_dims:
                raise ValueError('Check that your actuation function dims align with your BDQ.')
            self._checked_action_dims = True

    def _action_framework_copy(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=0)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            n_zones = self.dqn_model.action_branches

            # -- Handle Different Action-Space Architectures --
            # BDQ-Based model
            if self.run.model == 3:
                # Inference from model
                action = self.action
            # DQN-Based model
            else:
                # Inference from model
                action_options = ''.join([str(action) for action in list(range(self.actuation_dim))])
                action_permutations = \
                    [[int(action) for action in seq] for seq in itertools.product(action_options, repeat=n_zones)]
                action = action_permutations[self.action]

            # -- Handle RNN Sequence --
            if self.run.rnn and not self.rnn_start:
                # Default action before enough sequence
                action = [1] * self.dqn_model.action_branches  # Stay
                self.action = action  # save as agent action


            action_cmd_print = {}
            for zone in range(self.dqn_model.action_branches):
                # Get zone details
                zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value
                zone_action = action[zone]

                """
                Actuation Encoding Here
                """
                heating_sp = 0
                cooling_sp = 0

                self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone} ({action_cmd_print[action]}):'
                          f' Zn Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def action_directory(self, action_id: int):
        """This function is used to select and pass the chosen actuation function. This allows it to be a parameter."""

        action_directory = {
            1: self.act_heat_cool_off_1,
            2: self.act_strict_setpoints_2,
            3: self.act_step_strict_setpoints_3,
            4: self.act_default_adjustments_4,
            5: self.act_cool_only_5,
            6: self.act_cool_only_default_adjustment_6,
            7: self.act_cool_only_on_off_7,
            8: self.act_cool_only_on_off_stay_8
        }

        return action_directory[action_id]

    # ----------------------------------------------- ACTION ENCODINGS ------------------------------------------------

    def act_cool_only_on_off_stay_8(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=3)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            n_zones = self.dqn_model.action_branches

            # -- Handle Different Action-Space Architectures --
            # BDQ-Based model
            if self.run.model == 3:
                # Inference from model
                action = self.action
            # DQN-Based model
            else:
                # Inference from model
                action_options = ''.join([str(action) for action in list(range(self.actuation_dim))])
                action_permutations = \
                    [[int(action) for action in seq] for seq in itertools.product(action_options, repeat=n_zones)]
                action = action_permutations[self.action]

            # -- Handle RNN Sequence --
            if self.run.rnn and not self.rnn_start:
                # Default action before enough sequence
                action = [1] * self.dqn_model.action_branches  # Stay
                self.action = action  # save as agent action

            action_cmd_print = {0: 'OFF', 1: 'COOL', 2: 'STAY', None: 'Availability OFF'}
            for zone in range(n_zones):
                # Get zone details
                zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value
                zone_action = action[zone]

                # Adjust thermostat setpoints per zone
                if zone_action == 0:
                    # OFF
                    cooling_sp = zone_temp + self.temp_buffer
                elif zone_action == 1:
                    # COOL
                    cooling_sp = zone_temp - self.temp_buffer
                    # Hold minimum cooling setpoint
                    if cooling_sp < 18:
                        cooling_sp = 18
                elif zone_action == 2:
                    # STAY
                    close_enough = 0.25  # deg C buffer (0.45 f)
                    # Manage when close enough to boundary, set at boundary to avoid unwanted penalty
                    lower_ideal_temp = self.indoor_temp_ideal_range[0]
                    upper_ideal_temp = self.indoor_temp_ideal_range[1]
                    if lower_ideal_temp - close_enough <= zone_temp <= lower_ideal_temp + close_enough:
                        cooling_sp = lower_ideal_temp
                    elif upper_ideal_temp - close_enough <= zone_temp <= upper_ideal_temp + close_enough:
                        cooling_sp = upper_ideal_temp
                    else:
                        cooling_sp = zone_temp
                    # Hold minimum cooling setpoint
                    if cooling_sp < 18:
                        cooling_sp = 18

                else:
                    # HVAC Availability OFF
                    cooling_sp = zone_action  # None

                heating_sp = 15.56
                self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone} ({action_cmd_print[zone_action]}):'
                          f' Zn Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_cool_only_on_off_7(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=2)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            n_zones = self.dqn_model.action_branches

            # BDQ-Based model
            if self.run.model == 3:
                action = self.action
            # DQN-Based model
            else:
                action_options = ''.join([str(action) for action in list(range(self.actuation_dim))])
                action_permutations = \
                    [[int(action) for action in seq] for seq in itertools.product(action_options, repeat=n_zones)]
                action = action_permutations[self.action]

            action_cmd_print = {0: 'OFF', 1: 'COOL', None: 'Availability OFF'}
            for zone in range(n_zones):
                # Get zone details
                zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value
                zone_action = action[zone]

                # Adjust thermostat setpoints per zone
                if zone_action == 0:
                    # OFF
                    cooling_sp = zone_temp + self.temp_buffer
                elif zone_action == 1:
                    # COOL
                    cooling_sp = zone_temp - self.temp_buffer
                    # Hold minimum cooling setpoint
                    if cooling_sp < 18:
                        cooling_sp = 18
                else:
                    # HVAC Availability OFF
                    cooling_sp = zone_action  # None

                heating_sp = 15.56
                self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone} ({action_cmd_print[zone_action]}):'
                          f' Zn Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_cool_only_default_adjustment_6(self, actuate=True, exploit=False):
        """
               Action callback function:
               Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

               :return: actuation dictionary - EMS variable name (key): actuation value (value)
               """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=7)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            n_zones = self.dqn_model.action_branches

            # -- Handle Different Action-Space Architectures --
            # BDQ-Based model
            if self.run.model == 3:
                # Inference from model
                action = self.action
            # DQN-Based model
            else:
                # Inference from model
                action_options = ''.join([str(action) for action in list(range(self.actuation_dim))])
                action_permutations = \
                    [[int(action) for action in seq] for seq in itertools.product(action_options, repeat=n_zones)]
                action = action_permutations[self.action]

            # -- Handle RNN Sequence --
            if self.run.rnn and not self.rnn_start:
                # Default action before enough sequence
                action = [4] * self.dqn_model.action_branches  # Stay
                self.action = action  # save as agent action

            occupied_setpoints = {
                0: self.indoor_temp_ideal_range[1] + 1.5,
                1: self.indoor_temp_ideal_range[1],
                2: 23.2,
                3: 22.5,
                4: 21.8,
                5: self.indoor_temp_ideal_range[0],
                6: self.indoor_temp_ideal_range[0] - 1.5
            }

            unoccupied_setpoints = {
                0: self.indoor_temp_unoccupied_range[1] + 1.5,
                1: self.indoor_temp_unoccupied_range[1],
                2: 27.1,
                3: 24.84,
                4: 22.56,
                5: 20.28,
                6: 18
            }

            occupied = self.sim.get_ems_data(['hvac_operation_sched'])

            for zone in range(n_zones):
                # Get zone details
                zone_action = action[zone]

                if occupied:
                    cooling_sp = occupied_setpoints[zone_action]
                else:
                    cooling_sp = unoccupied_setpoints[zone_action]

                self.actuation_dict[f'zn{zone}_heating_sp'] = 15.56
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_cool_only_5(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=9)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            n_zones = self.dqn_model.action_branches

            # -- Handle Different Action-Space Architectures --
            # BDQ-Based model
            if self.run.model == 3:
                # Inference from model
                action = self.action
            # DQN-Based model
            else:
                # Inference from model
                action_options = ''.join([str(action) for action in list(range(self.actuation_dim))])
                action_permutations = \
                    [[int(action) for action in seq] for seq in itertools.product(action_options, repeat=n_zones)]
                action = action_permutations[self.action]

            # -- Handle RNN Sequence --
            if self.run.rnn and not self.rnn_start:
                # Default action before enough sequence
                action = [5] * self.dqn_model.action_branches  # Stay
                self.action = action  # save as agent action

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            # cooling_setpoints = {
            #     0: 18.1,
            #     1: 19.1,
            #     2: 21.1,  # LB Comfort
            #     3: 22,
            #     4: 23,
            #     5: 23.89,  # UB Comfort
            #     6: 25.72,
            #     7: 27.56,
            #     8: 29.4,
            #     9: 31
            # }
            cooling_setpoints = {
                0: 19.1,
                1: 21.1,  # LB Comfort
                2: 21.8,
                3: 22.5,
                4: 23.2,
                5: 23.89,  # UB Comfort
                6: 24.5,
                7: 25.5,
                8: 26.5
            }

            action_cmd_print = cooling_setpoints
            for zone in range(self.dqn_model.action_branches):
                zone_action = action[zone]
                zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value

                cooling_sp = cooling_setpoints[zone_action]
                # heating_sp = 21.1 if cooling_sp - 21.1 > self.setpoint_deadband else cooling_sp - self.setpoint_deadband
                heating_sp = 15.56

                self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone} ({action_cmd_print[zone_action]}):'
                          f' Zn Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_default_adjustments_4(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=6)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            occupied_setpoints = {
                0: [21.1, 23.89],  # IDEAL
                1: [19.1, 21.1],  # LOWER
                2: [21.1, 22.5],  # INNER LOWER
                3: [22.5, 23.89],  # INNER UPPER
                4: [23.89, 25.89],  # UPPER
                5: [21.8, 23.19]  # INNER
            }

            unoccupied_setpoints = {
                0: [15.56, 29.4],  # IDEAL
                1: [13.56, 15.56],  # LOWER
                2: [15.56, 22.5],  # INNER LOWER
                3: [22.5, 29.4],  # INNER UPPER
                4: [29.4, 32.4],  # UPPER
                5: [21.1, 23.89]  # INNER
            }

            occupied = self.sim.get_ems_data(['hvac_operation_sched'])

            for zone_i, action in enumerate(self.action):
                if occupied:
                    heating_sp, cooling_sp = occupied_setpoints[action]
                else:
                    heating_sp, cooling_sp = unoccupied_setpoints[action]

                self.actuation_dict[f'zn{zone_i}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone_i}_cooling_sp'] = cooling_sp

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_step_strict_setpoints_3(self, actuate=True, exploit=False):
        """
        Action callback function:
        Step up/down/nothing between fixed set of setpoint bounds

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """
        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=3)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            action_cmd_print = {0: 'STAY', 1: 'UP', 2: 'DOWN', None: 'Availability OFF'}

            setpoint_windows = {
                0: [13, 15.56],
                1: [15.56, 17],  # LB
                2: [17, 21.1],
                3: [21.1, 23.89],  # comfort
                4: [23.89, 27],
                5: [27, 29.4],  # UB
                6: [29.4, 32]
            }

            current_setpoints = self.current_setpoint_windows
            for zone_i, action in enumerate(self.action):

                if action == 1 and list(setpoint_windows.keys())[-1] != current_setpoints[zone_i]:
                    # UP SETPOINT
                    current_setpoints[zone_i] += 1
                elif action == 2 and 0 != current_setpoints[zone_i]:
                    # DOWN SETPOINT
                    current_setpoints[zone_i] -= 1
                else:
                    # STAY, or @ Boudaries
                    current_setpoints[zone_i] += 0

                # Set heating/cooling setpoints from fixed windows
                heating_sp = setpoint_windows[current_setpoints[zone_i]][0]
                cooling_sp = setpoint_windows[current_setpoints[zone_i]][1]
                self.actuation_dict[f'zn{zone_i}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone_i}_cooling_sp'] = cooling_sp

                if self._print:
                    zone_temp = self.mdp.ems_master_list[f'zn{zone_i}_temp'].value
                    print(f'\t\tZone{zone_i} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_strict_setpoints_2(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """
        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=6)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO THERMOSTAT SETPOINTS --
            action_cmd_print = {0: 'LOWEST', 1: 'LOWER', 2: 'IDEAL', 3: 'HIGHER', 4: 'HIGHEST'}

            for zone_i, action in enumerate(self.action):
                zone_temp = self.mdp.ems_master_list[f'zn{zone_i}_temp'].value

                actuation_cmd_dict = {
                    0: [15.1, 18.1],  # LOWEST
                    1: [18.1, 21.1],  # LOWER
                    2: [21.1, 23.9],  # IDEAL*
                    3: [23.9, 26.9],  # HIGHER
                    4: [26.9, 29.9],  # HIGHEST
                    5: [15.1, 29.9],  # OUTER
                }

                heating_sp = actuation_cmd_dict[action][0]
                cooling_sp = actuation_cmd_dict[action][1]

                self.actuation_dict[f'zn{zone_i}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone_i}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone_i} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')
        else:
            # Offline Learning
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    def act_heat_cool_off_1(self, actuate=True, exploit=False):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        # Check action space dim aligns with created BDQ
        self._action_dimension_check(this_actuation_functions_dims=3)

        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self._explore_exploit_process(exploit)

            # -- ENCODE ACTIONS TO HVAC COMMAND --
            action_cmd_print = {0: 'OFF', 1: 'HEAT', 2: 'COOL', None: 'Availability OFF'}
            for zone, action in enumerate(self.action):
                zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value

                # if all((self.indoor_temp_limits - zone_temp) < 0) or \
                #         all((self.indoor_temp_ideal_range - zone_temp) > 0):
                #     # outside safe comfortable bounds
                #     # print('unsafe temps')
                #     pass

                # adjust thermostat setpoints accordingly
                if action == 0:
                    # OFF
                    heating_sp = zone_temp - self.temp_deadband / 2
                    cooling_sp = zone_temp + self.temp_deadband / 2
                elif action == 1:
                    # HEAT
                    heating_sp = zone_temp + self.temp_buffer
                    cooling_sp = zone_temp + self.temp_buffer + self.temp_deadband
                elif action == 2:
                    # COOL
                    heating_sp = zone_temp - self.temp_buffer - self.temp_deadband
                    cooling_sp = zone_temp - self.temp_buffer
                else:
                    # HVAC Availability OFF
                    heating_sp = action  # None
                    cooling_sp = action  # None

                self.actuation_dict[f'zn{zone}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone}_cooling_sp'] = cooling_sp

                if self._print:
                    print(f'\t\tZone{zone} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                          f' Heating Sp = {round(heating_sp, 2)},'
                          f' Cooling Sp = {round(cooling_sp, 2)}')

        # Offline Learning
        else:
            pass

        # Combine system actuations with aux actions
        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict

    # ------------------------------------------------- REWARD -------------------------------------------------
    def _reward4(self):
        """SPARSE Reward function - per component, per zone."""

        n_zones = self.dqn_model.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': np.array([0, 0]) for zone_i in range(n_zones)}

        if (self.current_step + 1) % self.run.reward_sparsity_ts == 0 and self.current_step > 0:
            lambda_comfort = 1
            lambda_rtp = self.run.lambda_rtp

            # -- GET DATA SINCE LAST INTERACTION --
            interaction_span = range(self.observation_frequency * self.run.reward_sparsity_ts)
            # ALL
            building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
            # COMFORT
            # Get comfortable temp bounds based on building hours - occupied vs. unoccupied
            temp_bounds = np.asarray([self.indoor_temp_ideal_range if building_hours[i] == 1
                                      else self.indoor_temp_unoccupied_range for i in interaction_span])
            # $RTP
            rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

            # Per Controlled Zone
            for zone_i, _ in reward_components_per_zone_dict.items():
                reward_per_component = np.array([])

                """
                 -- COMFORTABLE TEMPS --
                For each zone, and array of minute interactions, each temperature is compared with the comfortable
                temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
                temperature from the nearest comfortable bound will be accounted for the reward.
                """
                zone_temps_since_last_interaction = np.asarray(
                    self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))
                # Get Temps Below
                too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                             zone_temps_since_last_interaction)
                temp_bounds_cold = temp_bounds[too_cold_temps != 0]  # get only lower bound temps where too cold
                too_cold_temps = too_cold_temps[too_cold_temps != 0]  # get only too cold zone temps
                # Get Temps Above
                too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                             zone_temps_since_last_interaction)
                temp_bounds_warm = temp_bounds[too_warm_temps != 0]  # get only lower bound temps where too cold
                too_warm_temps = too_warm_temps[too_warm_temps != 0]  # get only too cold zone temps

                # MSE penalty for temps above and below comfortable bounds
                reward = - (abs(too_cold_temps - temp_bounds_cold[:, 0]) ** self.run.comfort_p_norm).sum() \
                         - (abs(too_warm_temps - temp_bounds_warm[:, 1]) ** self.run.comfort_p_norm).sum()
                reward *= lambda_comfort
                reward_per_component = np.append(reward_per_component, reward)

                """
                 -- DR, RTP $ --
                For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
                heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
                timestep will be accounted for * the normalized HVAC energy usage
                """
                # Cooling
                cooling_energy_since_last_interaction = np.asarray(
                    self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
                ) / hvac_electricity_energy[f'{zone_i}_cooling_electricity_max']

                fan_electricity_since_last_interaction = np.asarray(
                    self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
                ) / hvac_electricity_energy[f'{zone_i}_fan_electricity_max']

                # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
                # fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

                # cooling_energy = fan_electricity_off_hours + cooling_energy_since_last_interaction
                cooling_energy = fan_electricity_since_last_interaction + cooling_energy_since_last_interaction

                # Timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
                cooling_factor = 1

                # Account for Cooling & Heating Costs
                cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy, rtp_since_last_interaction)

                # reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()
                reward = cooling_timesteps_cost.sum()
                reward *= lambda_rtp
                reward_per_component = np.append(reward_per_component, reward)

                """
                All Reward Components / Zone
                """
                reward_components_per_zone_dict[zone_i] = reward_per_component

                if self._print:
                    print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    def _reward3(self):
        """Reward function - per component, per zone."""

        lambda_comfort = 1
        lambda_rtp = self.run.lambda_rtp

        n_zones = self.dqn_model.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        interaction_span = range(self.observation_frequency)
        # ALL
        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        # COMFORT
        # Get comfortable temp bounds based on building hours - occupied vs. unoccupied
        temp_bounds = np.asarray([self.indoor_temp_ideal_range if building_hours[i] == 1
                                  else self.indoor_temp_unoccupied_range for i in interaction_span])
        # $RTP
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

        # Per Controlled Zone
        for zone_i, reward_list in reward_components_per_zone_dict.items():
            reward_per_component = np.array([])

            """
             -- COMFORTABLE TEMPS --
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))
            # Get Temps Below
            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]  # get only lower bound temps where too cold
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # get only too cold zone temps
            # Get Temps Above
            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]  # get only lower bound temps where too cold
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # get only too cold zone temps

            # MSE penalty for temps above and below comfortable bounds
            # reward = - ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() \
            #          - ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()
            reward = - (abs(too_cold_temps - temp_bounds_cold[:, 0])).sum() \
                     - (abs(too_warm_temps - temp_bounds_warm[:, 1])).sum()
            reward *= lambda_comfort

            reward_per_component = np.append(reward_per_component, reward)

            """
             -- DR, RTP $ --
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for * the normalized HVAC energy usage
            """
            # Cooling
            cooling_energy_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            ) / hvac_electricity_energy[f'{zone_i}_cooling_electricity_max']

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            ) / hvac_electricity_energy[f'{zone_i}_fan_electricity_max']

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            # fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            # cooling_energy = fan_electricity_off_hours + cooling_energy_since_last_interaction
            cooling_energy = fan_electricity_since_last_interaction + cooling_energy_since_last_interaction

            # Timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
            cooling_factor = 1

            # Account for Cooling & Heating Costs
            cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy, rtp_since_last_interaction)

            # reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()
            reward = cooling_timesteps_cost.sum()
            reward *= lambda_rtp

            reward_per_component = np.append(reward_per_component, reward)

            """
            All Reward Components / Zone
            """
            reward_components_per_zone_dict[zone_i] = reward_per_component

            if self._print:
                print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    def _reward2(self):
        """Reward function - per component, per zone."""

        lambda_comfort = 1
        lambda_rtp = 0.03 * 1
        lambda_intermittent = 1

        n_zones = self.dqn_model.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        interaction_span = range(self.observation_frequency)
        # ALL
        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        # COMFORT
        # Get comfortable temp bounds based on building hours - occupied vs. unoccupied
        temp_bounds = np.asarray([self.indoor_temp_ideal_range if building_hours[i] == 1
                                  else self.indoor_temp_unoccupied_range for i in interaction_span])
        # $RTP
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))
        # WIND
        wind_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['wind_gen'], interaction_span))
        total_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['total_gen'], interaction_span))

        # Per Controlled Zone
        for zone_i, reward_list in reward_components_per_zone_dict.items():
            reward_per_component = np.array([])

            """
             -- COMFORTABLE TEMPS --
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))
            # Get Temps Below
            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]  # get only lower bound temps where too cold
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # get only too cold zone temps
            # Get Temps Above
            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]  # get only lower bound temps where too cold
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # get only too cold zone temps

            # MSE penalty for temps above and below comfortable bounds
            # reward = - ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() \
            #          - ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()
            reward = - (abs(too_cold_temps - temp_bounds_cold[:, 0])).sum() \
                     - (abs(too_warm_temps - temp_bounds_warm[:, 1])).sum()
            reward *= lambda_comfort

            reward_per_component = np.append(reward_per_component, reward)

            """
             -- DR, RTP $ --
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for * the normalized HVAC energy usage
            """
            # Heating
            # heating_electricity_since_last_interaction = np.asarray(
            #     self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            # ) / hvac_electricity_energy[f'{zone_i}_heating_electricity_max']
            # heating_gas_since_last_interaction = np.asarray(
            #     self.sim.get_ems_data([f'{zone_i}_heating_gas'], interaction_span))
            # Cooling
            cooling_energy_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            ) / hvac_electricity_energy[f'{zone_i}_cooling_electricity_max']

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            ) / hvac_electricity_energy[f'{zone_i}_fan_electricity_max']

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            # heating_energy = fan_electricity_off_hours + heating_electricity_since_last_interaction  # + heating_gas_since_last_interaction
            cooling_energy = fan_electricity_off_hours + cooling_energy_since_last_interaction

            # Timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
            cooling_factor = 1
            heating_factor = 1

            # Account for Cooling & Heating Costs
            cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy, rtp_since_last_interaction)
            # heating_timesteps_cost = - heating_factor * np.multiply(heating_energy, rtp_since_last_interaction)

            # reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()
            reward = (cooling_timesteps_cost).sum()
            reward *= lambda_rtp

            reward_per_component = np.append(reward_per_component, reward)

            """
            -- INTERMITTENT RENEWABLE ENERGY USAGE --
            Quantify how much HVAC electricity consumed came from wind and total.
            """
            # total_hvac_electricity = heating_electricity_since_last_interaction \
            #                          + cooling_energy_since_last_interaction \
            #                          + fan_electricity_off_hours
            #
            # # joules_to_MWh = 2.77778e-10
            # # total_hvac_electricity = joules_to_MWh * total_hvac_electricity
            # intermittent_gen_mix = np.divide(wind_gen_since_last_interaction, total_gen_since_last_interaction)
            # # Normalize, stretch to cover 0-1 better, range for 2019 is 0-~0.6
            # intermittent_gen_mix = ((intermittent_gen_mix - 0) / (0.7 - 0)) * (1 - 0.0) + 0  # min/max
            #
            # reward = np.multiply(total_hvac_electricity, (intermittent_gen_mix - 1)).sum()
            # reward *= lambda_intermittent
            #
            # reward_per_component = np.append(reward_per_component, reward)

            """
            All Reward Components / Zone
            """
            reward_components_per_zone_dict[zone_i] = reward_per_component

            if self._print:
                print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    def _reward1(self):
        """Reward function - per component, per zone."""

        lambda_comfort = 0.2
        lambda_rtp = 0.005
        lambda_intermittent = 1000

        n_zones = self.dqn_model.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        interaction_span = range(self.observation_frequency)
        # ALL
        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        # COMFORT
        # get comfortable temp bounds based on building hours - occupied vs. unoccupied
        temp_bounds = [self.indoor_temp_ideal_range if building_hours[i] == 1 else self.indoor_temp_unoccupied_range
                       for i in interaction_span]
        temp_bounds = np.asarray(temp_bounds)
        # $RTP
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))
        # WIND
        wind_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['wind_gen'], interaction_span))
        total_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['total_gen'], interaction_span))

        # Per Controlled Zone
        for zone_i, reward_list in reward_components_per_zone_dict.items():
            reward_per_component = np.array([])

            """
             -- COMFORTABLE TEMPS --
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))

            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # only cold temps left

            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # only warm temps left

            # MSE penalty for temps above and below comfortable bounds
            reward = - ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() \
                     - ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()
            reward *= lambda_comfort

            reward_per_component = np.append(reward_per_component, reward)

            """
             -- DR, RTP $ --
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for. Note, that this is not multiplied by the energy used, such that this reward
            is agnostic to the zone size and incident load. 
            """

            heating_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            )
            # heating_gas_since_last_interaction = np.asarray(
            #     self.sim.get_ems_data([f'{zone_i}_heating_gas'], interaction_span))

            cooling_energy_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            )

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            heating_energy = fan_electricity_off_hours + heating_electricity_since_last_interaction  # + heating_gas_since_last_interaction
            cooling_energy = fan_electricity_off_hours + cooling_energy_since_last_interaction

            # Timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
            cooling_factor = 1
            heating_factor = 1

            # Only account for heating or cooling actions
            cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy > heating_energy,
                                                                    rtp_since_last_interaction)
            heating_timesteps_cost = - heating_factor * np.multiply(heating_energy > cooling_energy,
                                                                    rtp_since_last_interaction)

            reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()
            reward *= lambda_rtp

            reward_per_component = np.append(reward_per_component, reward)

            """
            -- INTERMITTENT RENEWABLE ENERGY USAGE --
            Quantify how much HVAC electricity consumed came from wind and total.
            """
            # TODO need to find a way to normalize energy usage between different sized zones
            total_hvac_electricity = heating_electricity_since_last_interaction \
                                     + cooling_energy_since_last_interaction \
                                     + fan_electricity_off_hours

            joules_to_MWh = 2.77778e-10
            total_hvac_electricity = joules_to_MWh * total_hvac_electricity
            intermittent_gen_mix = np.divide(wind_gen_since_last_interaction, total_gen_since_last_interaction)

            reward = np.multiply(total_hvac_electricity, intermittent_gen_mix - 1).sum()
            reward *= lambda_intermittent

            reward_per_component = np.append(reward_per_component, reward)

            """
            All Reward Components / Zone
            """
            reward_components_per_zone_dict[zone_i] = reward_per_component

            if self._print:
                print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    # ------------------------------------------------- RESULTS -------------------------------------------------
    # IN USE
    def _get_total_reward(self, aggregate_type: str):
        """Aggregates value from reward dict organized by zones and reward components"""

        # DQN-style model
        if self.run.model == 1 or self.run.model == 2:
            if aggregate_type == 'sum':
                return np.array(list(self.reward_dict.values())).sum()
            elif aggregate_type == 'mean':
                return np.array(list(self.reward_dict.values())).mean()

        # BDQ-style model
        elif self.run.model == 3:
            if self.run.combine_reward:
                # Get single scalar
                if aggregate_type == 'sum':
                    return np.array(list(self.reward_dict.values())).sum()
                elif aggregate_type == 'mean':
                    return np.array(list(self.reward_dict.values())).mean()
            else:
                # Keep reward separate per zone
                if aggregate_type == 'sum':
                    return np.array(list(self.reward_dict.values())).sum(axis=1)
                elif aggregate_type == 'mean':
                    return np.array(list(self.reward_dict.values())).mean(axis=1)

    # IN USE
    def _get_comfort_results(self):
        """
        For each timestep, and each zone, we calculate the weighted sum of temps outside the comfortable bounds.
        This represents a comfort compliance metric.
        :return: comfort compliance metric. A value of dissatisfaction. 0 is optimal.
        """

        n_zones = self.dqn_model.action_branches
        interaction_span = range(self.observation_frequency)
        controlled_zone_names = [f'zn{zone_i}' for zone_i in range(n_zones)]

        # Temp Bounds
        temp_schedule = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        temp_bounds = [self.indoor_temp_ideal_range if temp_schedule[i] == 1 else self.indoor_temp_unoccupied_range
                       for i in interaction_span]
        temp_bounds = np.asarray(temp_bounds)

        # Per Controlled Zone
        uncomfortable_metric = 0
        for zone_i in controlled_zone_names:
            # -- COMFORTABLE TEMPS --
            """
            For each zone, and array of minute interactions, each temperature is compared with the comfortable
            temperature bounds for the given timestep. If temperatures are out of bounds, the (-) MSE of that
            temperature from the nearest comfortable bound will be accounted for the reward.
            """
            zone_temps_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_temp'], interaction_span))

            # Get only temps colder than idea
            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # only cold temps left
            cold_temp_difference = too_cold_temps - temp_bounds_cold[:, 0]

            # Get only temps warmer than ideal
            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # only warm temps left
            warm_temp_difference = too_warm_temps - temp_bounds_warm[:, 1]

            # MSE penalty for temps above and below comfortable bounds
            uncomfortable_metric += (cold_temp_difference ** 2).sum() + \
                                    (warm_temp_difference ** 2).sum()

            # Histogram data
            self.cold_temps_histogram_data = np.append(self.cold_temps_histogram_data, cold_temp_difference)
            self.warm_temps_histogram_data = np.append(self.warm_temps_histogram_data, warm_temp_difference)

        if self._print:
            print(f'\n\tComfort: {round(uncomfortable_metric, 2)}, '
                  f'Cumulative: {round(self.comfort_dissatisfaction_total + uncomfortable_metric, 2)}')

        return uncomfortable_metric

    # IN USE
    def _get_rtp_hvac_cost_and_wind_results(self):
        """
        - $RTP -
        For each timestep, and each zone, we calculate the cost of HVAC electricity use based on RTP.
        This represents a DR compliance and monetary cost metric.
        - Energy Generation -
        For each timestep, and each zone, we calculate the ratio of wind energy to total energy generated and
        see how much HVAC energy uses.
        This represents how much our HVAC utilized wind energy throughout the year.

        :return: monetary cost of HVAC per interaction span metric. $0 is optimal.
        """

        n_zones = self.dqn_model.action_branches
        interaction_span = range(self.observation_frequency)
        controlled_zone_names = [f'zn{zone_i}' for zone_i in range(n_zones)]

        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)

        # RTP of last X timesteps
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

        # Energy Generation of last X timesteps
        wind_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['wind_gen'], interaction_span))
        total_gen_since_last_interaction = np.asarray(self.sim.get_ems_data(['total_gen'], interaction_span))

        # Per Controlled Zone
        rtp_hvac_costs = 0
        for zone_i in controlled_zone_names:
            # -- DR, RTP $ --
            """
            - RTP -
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for proportional to the energy used.
            - Wind -
            For each timestep, and each zone, we calculate the ratio of wind energy to total energy generated and
            see how much HVAC energy uses.  
            """

            # heating_electricity = np.asarray(
            #     self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            # )

            cooling_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            )

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            joules_to_MWh = 2.77778e-10
            # total_hvac_electricity = joules_to_MWh * \
            #                          (heating_electricity + cooling_electricity + fan_electricity_off_hours)
            total_hvac_electricity = joules_to_MWh * \
                                     (cooling_electricity + fan_electricity_off_hours)

            # timestep-wise RTP cost, accounting for HVAC electricity usage
            hvac_electricity_costs = np.multiply(total_hvac_electricity, rtp_since_last_interaction)
            rtp_hvac_costs += hvac_electricity_costs.sum()

            # -- RTP Histogram - collect rtp prices when a zone is heating/cooling
            # rtp_hvac_usage = rtp_since_last_interaction[heating_electricity != cooling_electricity]
            # self.rtp_histogram_data.extend(list(rtp_hvac_usage))  # get rid of 0s for when heat/cool OFF

            # -- Renewable Energy
            # how much HVAC energy used came from wind and total
            wind_energy_hvac_usage = np.multiply(total_hvac_electricity, wind_gen_since_last_interaction).sum()
            total_energy_hvac_usage = np.multiply(total_hvac_electricity, total_gen_since_last_interaction).sum()
            self.wind_energy_hvac_data.append(wind_energy_hvac_usage)
            self.total_energy_hvac_data.append(total_energy_hvac_usage)

        if self._print:
            print(
                f'\tRTP: ${round(rtp_hvac_costs, 2)}, Cumulative: ${round(self.hvac_rtp_costs_total + rtp_hvac_costs, 2)}')

        return rtp_hvac_costs

    def _get_rtp_hvac_cost_results(self):
        """
        For each timestep, and each zone, we calculate the cost of HVAC electricity use based on RTP.
        This represents a DR compliance and monetary cost metric.
        :return: monetary cost of HVAC per interaction span metric. $0 is optimal.
        """

        n_zones = self.dqn_model.action_branches
        interaction_span = range(self.observation_frequency)
        controlled_zone_names = [f'zn{zone_i}' for zone_i in range(n_zones)]

        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)

        # RTP of last X timesteps
        rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))

        # Per Controlled Zone
        rtp_hvac_costs = 0
        for zone_i in controlled_zone_names:
            # -- DR, RTP $ --
            """
            For each zone, and array of minute interactions, heating and cooling energy will be compared to see if HVAC
            heating, cooling, or Off actions occurred per each timestep. If heating or cooling occurs, the -$RTP for the
            timestep will be accounted for proportional to the energy used.
            """

            # heating_electricity = np.asarray(
            #     self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            # )

            cooling_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            )

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            # fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            joules_to_MWh = 2.77778e-10
            total_hvac_electricity = joules_to_MWh * (cooling_electricity + fan_electricity_since_last_interaction)

            # timestep-wise RTP cost, accounting for HVAC electricity usage
            hvac_electricity_costs = np.multiply(total_hvac_electricity, rtp_since_last_interaction)
            rtp_hvac_costs += hvac_electricity_costs.sum()

            # RTP Histogram - collect rtp prices when a zone is heating/cooling
            # rtp_hvac_usage = rtp_since_last_interaction[heating_electricity != cooling_electricity]
            # get rid of 0s for when heat/cool OFF
            # self.rtp_histogram_data.extend(list(rtp_hvac_usage))
            # self.rtp_histogram_data.append(list(zip(rtp_since_last_interaction, total_hvac_electricity)))

        # print(
        #     f'\tRTP: ${round(rtp_hvac_costs, 2)}, Cumulative: ${round(self.hvac_rtp_costs_total + rtp_hvac_costs, 2)}')

        return rtp_hvac_costs
