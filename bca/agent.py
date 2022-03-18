import time
import numpy as np

import torch

from .bdq import BranchingDQN, EpsilonGreedyStrategy, ReplayMemory

from emspy import BcaEnv, MdpManager


class Agent:
    def __init__(self,
                 sim: BcaEnv,
                 mdp: MdpManager,
                 dqn_model: BranchingDQN,
                 policy: EpsilonGreedyStrategy,
                 replay_memory: ReplayMemory,
                 interaction_frequency: int,
                 learning_loop: int = 1
                 ):

        # -- SIMULATION STATES --
        self.sim = sim
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
        self.action = None
        self.actuation_dict = {}
        self.epsilon = policy.start
        self.fixed_epsilon = None  # optional fixed exploration rate
        self.greedy_epsilon = policy
        # misc
        self.temp_deadband = 5  # distance between heating and cooling setpoints
        self.temp_buffer = 3  # new setpoint distance from current temps

        # -- REWARD --
        self.reward_dict = None
        self.reward = 0
        self.reward_sum = 0

        # -- CONTROL GOALS --
        self.indoor_temp_ideal_range = np.array([21.1, 23.89])  # occupied hours, based on OS model
        self.indoor_temp_unoccupied_range = np.array(
            [15.6 - 0.5, 29.4 + 0.5])  # mimic night cycle manager, + 1/2 temp tolerance
        self.indoor_temp_limits = np.array([15, 30])  # ??? needed?

        # -- TIMING --
        self.interaction_frequency = interaction_frequency
        self.n_ts = 0
        self.current_step = 0

        # -- INTERACTION FREQUENCIES --
        self.observation_ts = 15  # how often agent will observe state & keep fixed action - off-policy
        self.action_ts = 15  # how often agent will observe state & act - on-policy
        self.action_delay = 15  # how many ts will agent be fixed at beginning of simulation

        # -- REPLAY MEMORY --
        self.memory = replay_memory

        # -- BDQ --
        self.bdq = dqn_model

        # -- PERFORMANCE RESULTS --
        self.comfort_dissatisfaction = 0
        self.hvac_rtp_costs = 0
        self.comfort_dissatisfaction_total = 0
        self.hvac_rtp_costs_total = 0

        # -- RESULTS TRACKING --
        self.rtp_histogram_data = []
        self.wind_energy_hvac_data = []
        self.total_energy_hvac_data = []

        # -- Misc. --
        self.learning = True
        self.learning_loop = learning_loop
        self.loss = 0
        self.once = True

    def observe(self, learn=True):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.sim.get_ems_data(['t_datetimes'])
        self.var_vals = self.mdp.update_ems_value_from_dict(self.sim.get_ems_data(self.var_names, return_dict=True))
        self.meter_vals = self.mdp.update_ems_value_from_dict(self.sim.get_ems_data(self.meter_names, return_dict=True))
        self.weather_vals = self.mdp.update_ems_value_from_dict(
            self.sim.get_ems_data(self.weather_names, return_dict=True))

        self.termination = self._is_terminal()

        print(f'\n\n{str(self.time)}\n')  # \n\n\tVars: {vars}\n\tMeters: {meters}\n\tWeather: {weather}')

        # -- MODIFY STATE --
        meter_names = [meter for meter in self.meter_names if 'fan' not in meter]  # remove unwanted fan meters

        # -- GET ENCODING --
        self.var_encoded_vals = self.mdp.get_ems_encoded_values(self.var_names)
        self.meter_encoded_vals = self.mdp.get_ems_encoded_values(meter_names)
        self.weather_encoded_vals = self.mdp.get_ems_encoded_values(self.weather_names)

        # -- ENCODED STATE --
        self.next_state_normalized = np.array(list(self.var_encoded_vals.values()) +
                                              list(self.weather_encoded_vals.values()) +
                                              list(self.meter_encoded_vals.values()), dtype=float)

        # -- REWARD --
        self.reward_dict = self._reward()
        self.reward = self._get_total_reward('sum')  # aggregate 'mean' or 'sum'

        # -- STORE INTERACTIONS --
        if self.action is not None:  # after first action, enough data available
            self.memory.push(self.state_normalized, self.action, self.next_state_normalized,
                             self.reward, self.termination)  # <S, A, S', R, t> - push experience to Replay Memory

        # -- LEARN BATCH --
        if learn:
            if self.memory.can_provide_sample():  # must have enough interactions stored
                for i in range(self.learning_loop):
                    batch = self.memory.sample()
                    self.loss = float(self.bdq.update_policy(batch))  # batch learning

        # -- PERFORMANCE RESULTS --
        self.comfort_dissatisfaction = self._get_comfort_results()
        # self.hvac_rtp_costs = self._get_rtp_hvac_cost_results()
        self.hvac_rtp_costs = self._get_rtp_hvac_cost_and_wind_results()
        self.comfort_dissatisfaction_total += self.comfort_dissatisfaction
        self.hvac_rtp_costs_total += self.hvac_rtp_costs

        # -- UPDATE DATA --
        self.state_normalized = self.next_state_normalized
        self.reward_sum += self.reward
        self.current_step += 1

        # -- REPORTING --
        # self._report_time()  # time
        print(f'\n\t*Reward: {round(self.reward, 2)}, Cumulative: {round(self.reward_sum, 2)}')

        # -- DO ONCE --
        if self.once:
            self.state_var_names = self.var_names + self.weather_names + meter_names
            self.once = False

        # -- TRACK REWARD --
        return self.reward  # return reward for emspy pd.df tracking

    def _get_aux_actuation(self):
        """
        Used to manage auxiliary actuation (likely schedule writing) in one place.
        """
        return {
            # Data Tracking
            # -- Rewards --
            'reward': self.reward,
            'reward_cumulative': self.reward_sum,
            # -- Results Metric --
            # Comfort
            'comfort': self.comfort_dissatisfaction,
            'comfort_cumulative': self.comfort_dissatisfaction_total,
            # RTP
            'rtp_tracker': self.hvac_rtp_costs,
            'rtp_cumulative': self.hvac_rtp_costs_total,
            # Wind
            'wind_hvac_use': self.wind_energy_hvac_data[-1],
            'total_hvac_use': self.total_energy_hvac_data[-1],
            # -- Learning --
            # loss
            'loss': self.loss
        }

    def act_heat_cool_off(self):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        if False:  # mdp.ems_master_list['hvac_operation_sched'].value == 0:
            self.action = [None] * self.bdq.action_branches  # return actuation to E+
            action_type = 'Availability OFF'

        # -- EXPLOITATION vs EXPLORATION --
        else:
            self.epsilon = self.greedy_epsilon.get_exploration_rate(self.current_step, self.fixed_epsilon)
            if np.random.random() < self.epsilon:
                # Explore
                self.action = np.random.randint(0, 3, self.bdq.action_branches)
                action_type = 'Explore'
            else:
                # Exploit
                self.action = self.bdq.get_greedy_action(torch.Tensor(self.state_normalized).unsqueeze(1))
                action_type = 'Exploit'

        print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

        # -- ENCODE ACTIONS TO HVAC COMMAND --
        action_cmd_print = {0: 'OFF', 1: 'HEAT', 2: 'COOL', None: 'Availability OFF'}
        for zone, action in enumerate(self.action):
            zone_temp = self.mdp.ems_master_list[f'zn{zone}_temp'].value

            if all((self.indoor_temp_limits - zone_temp) < 0) or all((self.indoor_temp_ideal_range - zone_temp) > 0):
                # outside safe comfortable bounds
                # print('unsafe temps')
                pass

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

            print(f'\t\tZone{zone} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                  f' Heating Sp = {round(heating_sp, 2)},'
                  f' Cooling Sp = {round(cooling_sp, 2)}')

        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)  # combine

        return self.actuation_dict

    # IN USE
    def act_strict_setpoints(self, actuate=True):
        if actuate:
            # -- EXPLOITATION vs EXPLORATION --
            self.epsilon = self.greedy_epsilon.get_exploration_rate(self.current_step, self.fixed_epsilon)
            if np.random.random() < self.epsilon:
                # Explore
                self.action = np.random.randint(0, self.bdq.action_dim, self.bdq.action_branches)
                action_type = 'Explore'
            else:
                # Exploit
                self.action = self.bdq.get_greedy_action(torch.Tensor(self.state_normalized).unsqueeze(1))
                action_type = 'Exploit'

            print(f'\n\tAction: {self.action} ({action_type}, eps = {self.epsilon})')

            # -- ENCODE ACTIONS TO THERMOSTAT SETPOINTS --
            action_cmd_print = {0: 'LOWEST', 1: 'LOWER', 2: 'IDEAL', 3: 'HIGHER', 4: 'HIGHEST'}

            for zone_i, action in enumerate(self.action):
                zone_temp = self.mdp.ems_master_list[f'zn{zone_i}_temp'].value

                actuation_cmd_dict = {
                    0: [15.1, 18.1],  # LOWEST
                    1: [18.1, 21.1],  # LOWER
                    2: [21.1, 23.9],  # IDEAL*
                    3: [23.9, 26.9],  # HIGHER
                    4: [26.9, 29.9]  # HIGHEST
                }

                heating_sp = actuation_cmd_dict[action][0]
                cooling_sp = actuation_cmd_dict[action][1]

                self.actuation_dict[f'zn{zone_i}_heating_sp'] = heating_sp
                self.actuation_dict[f'zn{zone_i}_cooling_sp'] = cooling_sp

                print(f'\t\tZone{zone_i} ({action_cmd_print[action]}): Temp = {round(zone_temp, 2)},'
                      f' Heating Sp = {round(heating_sp, 2)},'
                      f' Cooling Sp = {round(cooling_sp, 2)}')

        aux_actuation = self._get_aux_actuation()
        self.actuation_dict.update(aux_actuation)  # combine/replace aux and control actuations

        return self.actuation_dict

    # IN USE
    def _reward(self):
        """Reward function - per component, per zone."""

        # TODO add some sort of normalization
        lambda_comfort = 0.1
        lambda_rtp = 0.01
        lambda_intermittant = 1

        n_zones = self.bdq.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        interaction_span = range(self.interaction_frequency)
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

            heating_energy = fan_electricity_since_last_interaction + heating_electricity_since_last_interaction  # + heating_gas_since_last_interaction
            cooling_energy = fan_electricity_since_last_interaction + cooling_energy_since_last_interaction

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
            total_hvac_electricity = heating_electricity_since_last_interaction + cooling_energy_since_last_interaction\
                                     + fan_electricity_since_last_interaction
            intermittent_gen_mix = np.divide(wind_gen_since_last_interaction, total_gen_since_last_interaction)

            reward = np.multiply(total_hvac_electricity, intermittent_gen_mix - 1)
            reward *= lambda_intermittant

            reward_per_component = np.append(reward_per_component, reward)

            """
            All Reward Components / Zone
            """
            reward_components_per_zone_dict[zone_i] = reward_per_component
            print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    def _reward_components_conditional(self, comfort=True, rtp=True, wind=True):
        """Reward function - per component, per zone."""
        # TODO

        # TODO add some sort of normalization
        lambda_comfort = 0.1
        lambda_rtp = 0.01
        lambda_intermittant = 1

        n_zones = self.bdq.action_branches
        reward_components_per_zone_dict = {f'zn{zone_i}': None for zone_i in range(n_zones)}

        # -- GET DATA SINCE LAST INTERACTION --
        interaction_span = range(self.interaction_frequency)
        # ALL
        building_hours = self.sim.get_ems_data(['hvac_operation_sched'], interaction_span)
        # COMFORT
        if comfort:
            # get comfortable temp bounds based on building hours - occupied vs. unoccupied
            temp_bounds = [self.indoor_temp_ideal_range if building_hours[i] == 1 else self.indoor_temp_unoccupied_range
                           for i in interaction_span]
            temp_bounds = np.asarray(temp_bounds)
        # $RTP
        if rtp:
            rtp_since_last_interaction = np.asarray(self.sim.get_ems_data(['rtp'], interaction_span))
        # WIND
        if wind:
            pass

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

            heating_energy = fan_electricity_since_last_interaction + heating_electricity_since_last_interaction  # + heating_gas_since_last_interaction
            cooling_energy = fan_electricity_since_last_interaction + cooling_energy_since_last_interaction

            # Timestep-wise RTP cost, not accounting for energy-usage, only that energy was used
            cooling_factor = 1
            heating_factor = 1

            # Only account for heating or cooling actions, Fan energy usage is implied here
            cooling_timesteps_cost = - cooling_factor * np.multiply(cooling_energy > heating_energy,
                                                                    rtp_since_last_interaction)
            heating_timesteps_cost = - heating_factor * np.multiply(heating_energy > cooling_energy,
                                                                    rtp_since_last_interaction)

            reward = (cooling_timesteps_cost + heating_timesteps_cost).sum()
            reward *= lambda_rtp

            reward_per_component = np.append(reward_per_component, reward)

            """
            # -- INTERMITTENT RENEWABLE ENERGY USAGE --
            TODO
            """

            # Reward Components per Zone
            reward_components_per_zone_dict[zone_i] = reward_per_component
            print(f'\tReward - {zone_i}: {reward_per_component}')

        return reward_components_per_zone_dict

    # IN USE
    def _get_total_reward(self, aggregate_type: str):
        """Aggregates value from reward dict organized by zones and reward components"""
        if aggregate_type == 'sum':
            return np.array(list(self.reward_dict.values())).sum()
        elif aggregate_type == 'mean':
            return np.array(list(self.reward_dict.values())).mean()

    # IN USE
    def _get_comfort_results(self):
        """
        For each timestep, and each zone, we calculate the weighted sum of temps outside the comfortable bounds.
        This represents a comfort compliance metric.
        :return: comfort compliance metric. A value of dissatisfaction. 0 is optimal.
        """

        n_zones = self.bdq.action_branches
        interaction_span = range(self.interaction_frequency)
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

            # get only temps colder than idea
            too_cold_temps = np.multiply(zone_temps_since_last_interaction < temp_bounds[:, 0],
                                         zone_temps_since_last_interaction)
            temp_bounds_cold = temp_bounds[too_cold_temps != 0]
            too_cold_temps = too_cold_temps[too_cold_temps != 0]  # only cold temps left

            # get only temps warmer than ideal
            too_warm_temps = np.multiply(zone_temps_since_last_interaction > temp_bounds[:, 1],
                                         zone_temps_since_last_interaction)
            temp_bounds_warm = temp_bounds[too_warm_temps != 0]
            too_warm_temps = too_warm_temps[too_warm_temps != 0]  # only warm temps left

            # MSE penalty for temps above and below comfortable bounds
            uncomfortable_metric += ((too_cold_temps - temp_bounds_cold[:, 0]) ** 2).sum() + \
                                    ((too_warm_temps - temp_bounds_warm[:, 1]) ** 2).sum()  # sum prev timestep

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

        n_zones = self.bdq.action_branches
        interaction_span = range(self.interaction_frequency)
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

            heating_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            )

            cooling_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            )

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            joules_to_MWh = 2.77778e-10
            total_hvac_electricity = joules_to_MWh * \
                                     (heating_electricity + cooling_electricity + fan_electricity_off_hours)

            # timestep-wise RTP cost, accounting for HVAC electricity usage
            hvac_electricity_costs = np.multiply(total_hvac_electricity, rtp_since_last_interaction)
            rtp_hvac_costs += hvac_electricity_costs.sum()

            # -- RTP Histogram - collect rtp prices when a zone is heating/cooling
            rtp_hvac_usage = rtp_since_last_interaction[heating_electricity != cooling_electricity]
            self.rtp_histogram_data.extend(list(rtp_hvac_usage))  # get rid of 0s for when heat/cool OFF

            # -- Renewable Energy
            # how much HVAC energy used came from wind and total
            wind_energy_hvac_usage = np.multiply(total_hvac_electricity, wind_gen_since_last_interaction).sum()
            total_energy_hvac_usage = np.multiply(total_hvac_electricity, total_gen_since_last_interaction).sum()
            self.wind_energy_hvac_data.append(wind_energy_hvac_usage)
            self.total_energy_hvac_data.append(total_energy_hvac_usage)

        print(
            f'\tRTP: ${round(rtp_hvac_costs, 2)}, Cumulative: ${round(self.hvac_rtp_costs_total + rtp_hvac_costs, 2)}')

        return rtp_hvac_costs

    def _get_rtp_hvac_cost_results(self):
        """
        For each timestep, and each zone, we calculate the cost of HVAC electricity use based on RTP.
        This represents a DR compliance and monetary cost metric.
        :return: monetary cost of HVAC per interaction span metric. $0 is optimal.
        """

        n_zones = self.bdq.action_branches
        interaction_span = range(self.interaction_frequency)
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

            heating_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_heating_electricity'], interaction_span)
            )

            cooling_electricity = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_cooling_electricity'], interaction_span)
            )

            fan_electricity_since_last_interaction = np.asarray(
                self.sim.get_ems_data([f'{zone_i}_fan_electricity'], interaction_span)
            )

            # Don't penalize for fan usage during day when it is REQUIRED for occupant ventilation, only Off-hours
            fan_electricity_off_hours = np.multiply(building_hours == 0, fan_electricity_since_last_interaction)

            joules_to_MWh = 2.77778e-10
            total_hvac_electricity = joules_to_MWh * \
                                     (heating_electricity + cooling_electricity + fan_electricity_off_hours)

            # timestep-wise RTP cost, accounting for HVAC electricity usage
            hvac_electricity_costs = np.multiply(total_hvac_electricity, rtp_since_last_interaction)
            rtp_hvac_costs += hvac_electricity_costs.sum()

            # RTP Histogram - collect rtp prices when a zone is heating/cooling
            rtp_hvac_usage = rtp_since_last_interaction[heating_electricity != cooling_electricity]
            # get rid of 0s for when heat/cool OFF
            self.rtp_histogram_data.extend(list(rtp_hvac_usage))

        print(
            f'\tRTP: ${round(rtp_hvac_costs, 2)}, Cumulative: ${round(self.hvac_rtp_costs_total + rtp_hvac_costs, 2)}')

        return rtp_hvac_costs

    def _is_terminal(self):
        """Determines whether the current state is a terminal state or not. Dictates TD update values."""
        return 0

    def _report_daily(self):
        self.time = self.sim.get_ems_data('t_datetimes')
        if self.time.day != self.prev_day and self.time.hour == 1:
            self.day_update = True
            print(f'{self.time.strftime("%m/%d/%Y")} - Trial: {self.trial} - Reward Daily Sum: '
                  f'{self.reward_sum - self.prior_reward_sum:0.4f}')
            print(f'Elapsed Time: {(time.time() - self.tictoc) / 60:0.2f} mins')
            # updates
            self.prior_reward_sum = self.reward_sum
            # update current/prev day
            self.prev_day = self.time.day
