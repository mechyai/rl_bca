"""
At this point (v3), the prototype BDQ is working with BDQN_v1.py:
- comfort MSE reward ONLY (=0 in comfortable bounds) throughout the entire night
- action encoding: thermostat steps via HEAT/COOL/OFF
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

import openstudio  # ver 3.2.0 !pip list
from emspy import emspy, idf_editor
from Current_Prototype import MDP_v1 as MDP
from Current_Prototype import BDQN_v1 as BDQ

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# -- FILE PATHS --
# E+ Download Path
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'
# IDF File / Modification Paths
idf_file_name = r'/BEM_5z_Prototype_PV_V2_AMY_alwayson/run/BEM_5z_V1.idf'
idf_final_file = r'A:/Files/PycharmProjects/RL-BCA/Current_Prototype/BEM/BEM_5z_V1.idf'
os_folder = r'A:/Files/PycharmProjects/RL-bca/Current_Prototype/BEM'
idf_file_base = os_folder + idf_file_name
# Weather Path
ep_weather_path = os_folder + r'/WeatherFiles/EPW/DallasTexas_2019CST.epw'
# Output .csv Path
cvs_output_path = ''


# -- INSTANTIATE MDP --
mdp = MDP.generate_mdp_from_tc()


# -- CUSTOM TRACKING --
data_tracking = {  # custom tracking for actuators, (handle + unit type)
    'reward': ('Schedule:Constant', 'Schedule Value', 'Reward Tracker', 'Dimensionless'),
    'reward_cumulative': ('Schedule:Constant', 'Schedule Value', 'Reward Cumulative', 'Dimensionless'),
    'wind_gen_relative': ('Schedule:Constant', 'Schedule Value', 'Wind Gen of Total', 'Dimensionless')
}
# link with ToC Actuators, remove unit types first
data_tracking_actuators = {}
for key, values in data_tracking.items():
    mdp.add_ems_element('actuator', key, values[0:3])  # exclude unit, leave handle


# -- Automated IDF Modification --
year = 2019
# create final file from IDF base
idf_editor.append_idf(idf_file_base, r'BEM/CustomIdfFiles/Automated/V1_IDF_modifications.idf', idf_final_file)
# daylight savings & holidays
# IDFeditor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/TEXAS_CST_Daylight_Savings_{year}.idf')
# add Schedule:Files
idf_editor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/ERCOT_RTM_{year}.idf')  # RTP
idf_editor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/ERCOT_FMIX_{year}_Wind.idf')  # FMIX, wind
idf_editor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/ERCOT_FMIX_{year}_Total.idf')  # FMIX, total
for h in range(12):  # DAM 12 hr forecast
    idf_editor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/ERCOT_DAM_12hr_forecast_{year}_{h}hr_ahead.idf')
# add Custom Meters
idf_editor.append_idf(idf_final_file, r'BEM/CustomIdfFiles/Automated/V1_custom_meters.idf')
# add Custom Data Tracking IDF Objs (reference ToC of Actuators)
for _, value in data_tracking.items():
    idf_editor.insert_custom_data_tracking(value[2], idf_final_file, value[3])

# -- Simulation Params --
cp = emspy.EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop
timesteps = 60


class Agent:
    def __init__(self, sim, dqn_model, policy, replay_memory, learning_loop: int = 1):
        # -- INTERACTION FREQUENCIES --
        self.observation_ts = 15  # how often agent will observe state & keep fixed action - off-policy
        self.action_ts = 15  # how often agent will observe state & act - on-policy
        self.action_delay = 15  # how many ts will agent be fixed at beginning of simulation

        # -- SIMULATION STATES --
        self.sim = sim
        # Get list of EMS objects
        self.vars = mdp.ems_type_dict['var']
        self.meters = mdp.ems_type_dict['meter']
        self.weather = mdp.ems_type_dict['weather']

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
        self.temp_buffer = 1  # new setpoint distance from current temps


        # -- REWARD --
        self.reward_sum = 0

        # -- CONTROL GOALS --
        self.indoor_temp_ideal_range = np.array([21.1, 23.89])  # occupied hours, based on OS model
        self.indoor_temp_unoccupied_range = np.array([15.6 + 0.5, 29.4 + 0.5])  # mimic night cycle manager, + 1/2 temp tolerance
        self.indoor_temp_limits = np.array([15, 30])  # ??? needed?

        # -- TIMING --
        self.n_ts = 0
        self.current_step = 0

        # -- REPLAY MEMORY --
        self.memory = replay_memory

        # -- BDQ --
        self.bdq = dqn_model

        # -- misc --
        self.learning = True
        self.learning_loop = learning_loop
        self.once = True

    def observe(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        time = self.sim.get_ems_data(['t_datetimes'])
        vars = mdp.update_ems_value(self.vars, self.sim.get_ems_data(mdp.get_ems_names(self.vars)))
        # meters = mdp.update_ems_value(self.meters, self.sim.get_ems_data(mdp.get_ems_names(self.meters)))
        weather = mdp.update_ems_value(self.weather,
                                       self.sim.get_ems_data(mdp.get_ems_names(self.weather)))

        print(f'\n\n{str(time)}')  # \n\n\tVars: {vars}\n\tMeters: {meters}\n\tWeather: {weather}')

        # -- ENCODING --
        self.next_state_normalized = np.array(list(vars.values()) + list(weather.values()), dtype=float)

        # -- ENCODED STATE --
        self.termination = self.is_terminal()

        # -- REWARD --
        self.reward = self.get_reward()

        # -- STORE INTERACTIONS --
        if self.action is not None:  # after first action, enough data available
            self.memory.push(self.state_normalized, self.action, self.next_state_normalized,
                             self.reward, self.termination)  # <S, A, S', R, t> - push experience to Replay Memory

        # -- LEARN BATCH --
        if self.learning:
            if self.memory.can_provide_sample():  # must have enough interactions stored
                # TODO handle control before training?
                for i in range(self.learning_loop):
                    batch = self.memory.sample()
                    self.bdq.update_policy(batch)  # batch learning

        # -- UPDATE DATA --
        self.state_normalized = self.next_state_normalized

        self.reward_sum += self.reward
        self.current_step += 1

        # -- REPORT --
        # self._report_time()  # time

        # -- DO ONCE --
        if self.once:
            self.state_var_names = list(vars.keys()) + list(weather.keys())
            self.once = False

        # -- REPORTING --
        print(f'\n\tReward: {round(self.reward, 2)}, Cumulative: {round(self.reward_sum, 2)}')

        # -- TRACK REWARD --
        return self.reward  # return reward for emspy pd.df tracking

    def act(self):
        """
        Action callback function:
        Takes action from network or exploration, then encodes into HVAC commands and passed into running simulation.

        :return: actuation dictionary - EMS variable name (key): actuation value (value)
        """

        if False: #mdp.ems_master_list['hvac_operation_sched'].value == 0:
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
        action_cmd = {0: 'OFF', 1: 'HEAT', 2: 'COOL', None: 'Availability OFF'}
        for zone, action in enumerate(self.action):
            zone_temp = mdp.ems_master_list[f'zn{zone}_temp'].value

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

            print(f'\t\tZone{zone} ({action_cmd[action]}): Temp = {round(zone_temp,2)},'
                  f' Heating Sp = {round(heating_sp,2)},'
                  f' Cooling Sp = {round(cooling_sp,2)}')

        aux_actuation = {
            # Data Tracking
            'reward': self.reward,
            'reward_cumulative': self.reward_sum,
        }
        # combine
        self.actuation_dict.update(aux_actuation)

        return self.actuation_dict
        # return aux_actuation

    def get_reward(self):
        """Reward function."""

        # TODO add some sort of normalization and lambda

        # -- COMFORTABLE TEMPS --BEM_EMS_v3.py
        if mdp.ems_master_list['hvac_operation_sched'].value == 0:
            # no HVAC availability

            aggregate_reward = 0
        else:
            # HVAC enabled
            reward_per_zone = np.array([])
            n_action = 5  # OR self.bdq.action_space
            for zone in range(n_action):  # for all zones
                zone_temp = mdp.ems_master_list[f'zn{zone}_temp'].value
                # outside ideal range
                if all((self.indoor_temp_ideal_range - zone_temp) < 0) or all((self.indoor_temp_ideal_range - zone_temp) > 0):
                    reward = min(self.indoor_temp_ideal_range - zone_temp) ** 2
                # inside ideal range
                else:
                    reward = 0
                reward_per_zone = np.append(reward_per_zone, reward)

            aggregate_reward = -reward_per_zone.mean()  # or SUM

        # -- DR, RTP $ --

        # -- RENEWABLE ENERGY --

        return aggregate_reward

    def is_terminal(self):
        """Determines whether the current state is a terminal state or not. Dictates TD update values."""
        return 0

    def _report_time(self):
        self.time = self.sim.get_ems_data('t_datetimes')
        if self.time.day != self.prev_day and self.time.hour == 1:
            self.day_update = True
            print(f'{self.time.strftime("%m/%d/%Y")} - Trial: {self.trial} - Reward Daily Sum: '
                  f'{self.reward_sum - self.prior_reward_sum:0.4f}')
            print(f'Elapsed Time: {(time.time() - self.tictoc)/60:0.2f} mins')
            # updates
            self.prior_reward_sum = self.reward_sum
            # update current/prev day
            self.prev_day = self.time.day

# Instantiate Model
act_obs_freq = 10  # timestep intervals



policy = BDQ.EpsilonGreedyStrategy(
            start=0.15,  # epsilon
            end=0.01,
            decay=0.00005
        )


exp_time = time.strftime("%Y%m%d_%H%M")
exp = f'Exp_{exp_time}'
os.mkdir(os.path.join('Tuning_Data', exp))  # make experiment folder

n_epochs = 3
exploit_epochs = {}

# batches_update = [(32,50),(32,150),(32,250),(32,250),(64,50),(64,150),(64,250),(64,250),(128,50),(128,150),(128,250),(128,250),(256,50),(256,150),(256,250),(256,250)]
batches_update = [(64,50),(64,150),(64,250),(128,50),(128,150),(128,250),(256,50),(256,150),(256,250)]
#
# batches_update = [(64, 100)]

for i, (batch_size, update_freq) in enumerate(batches_update):

    # change learning loop
    # loop = 1 if i < 10 else 2

    # if batches[epoch - 1] != batches[epoch]:
    if True:
        # create new model with each batch pair
        bdq_model = BDQ.BranchingDQN(
            observation_space=12,
            action_space=4,  # 5 building zones
            action_bins=3,  # heat/cool/off
            target_update_freq=update_freq,  # int(60 / ((60 / timesteps) * act_obs_freq) * 48),  # every 12 hrs
            learning_rate=0.0005,
            gamma=0.99,
            # architecture
            shared_hidden_dim1=128,
            shared_hidden_dim2=96,
            state_hidden_dim=64,
            action_hidden_dim=28,
            td_target='mean',  # mean or max
            gradient_clip_norm=5
        )

        experience_replay = BDQ.ReplayMemory(
            capacity=int(60 / ((60 / timesteps) * act_obs_freq) * 24 * 21),  # 21 days
            batch_size=batch_size
        )

    for epoch in range(n_epochs):
        time_start = time.time()

        # -- Create Building Energy Simulation Instance --
        sim = emspy.BcaEnv(ep_path, idf_final_file, timesteps,
                           mdp.tc_var, mdp.tc_intvar, mdp.tc_meter, mdp.tc_actuator, mdp.tc_weather)

        # -- Instantiate RL Agent --
        agent = Agent(sim, bdq_model, policy, experience_replay, learning_loop=1)

        # agent.fixed_epsilon = 0
        # agent.learning = False

        # -- Explore VS Exploit --
        # if i in exploit_epochs:
        #     agent.fixed_epsilon = 0
        #     agent.learning = False
        # else:
        #     agent.fixed_epsilon = None
        #     agent.learning = True

        hyperparameters_dict = {
            'target_update_freq': bdq_model.target_update_freq,  # every 12 hrs
            'learning_rate': bdq_model.learning_rate,
            'gamma': bdq_model.gamma,
            'start': policy.start,  # epsilon
            'end': policy.end,
            'decay': policy.decay,
            'capacity': experience_replay.capacity,
            'batch_size': experience_replay.batch_size,
            'learning_loops': agent.learning_loop,
            'sim_timesteps': timesteps,
            'action/obs freq': act_obs_freq,
            'gradient_clip_max': bdq_model.gradient_clip_norm
        }

        # Set Calling Point(s) & Callback Function(s)
        sim.set_calling_point_and_callback_function(cp, agent.observe, agent.act, True, act_obs_freq, act_obs_freq)

        # -- RUN SIM --
        sim.run_env(ep_weather_path)

        # -- GET SIM DFs --
        dfs = sim.get_df()
        dfs['reward']['cumulative'] = dfs['reward'][['reward']].cumsum()  # create cumulative reward column
        cumulative_reward = float(dfs['reward'][['cumulative']].iloc[-1])  # get final cumulative reward

        # -- RESULTS --
        # file names
        model_name = f'bdq_{time.strftime("%Y%m%d_%H%M")}.pt'
        folder = os.path.join('Tuning_Data', exp)
        file_path = os.path.join(folder, f'_bdq_report_{exp_time}.txt')
        plot_path = os.path.join(folder, model_name[:-3])

        # plot
        plt.figure()
        fig, ax = plt.subplots()
        dfs['reward'].plot(x='Datetime', y='reward', ax=ax)
        dfs['reward'].plot(x='Datetime', y='cumulative', ax=ax, secondary_y=True)
        plt.title(model_name[:-3] + f', batch_size={experience_replay.batch_size}, update_freq={bdq_model.target_update_freq}')

        # misc
        note = "testing new gradient_clipping..."

        # -- SAVE --
        # torch.save(bdq_model.policy_network, os.path.join(folder, model_name))  # save model
        fig.savefig(plot_path)
        plt.close('all')

        # -- OUTPUT --
        with open(file_path, 'a+') as file:
            file.write(f'\n\n\n\n Model Name: {model_name}')
            file.write(f'\nReward Plot Name: {model_name[:-3]}.png')
            file.write(f'\n\n\t*Epochs trained: {epoch}')
            file.write(f'\n\tTime Train: {round(time_start - time.time(), 2)/60} mins')
            file.write(f'\n\t******* Cumulative Reward: {cumulative_reward}')
            file.write(f'\n\tState Space: {agent.state_var_names}')
            file.write('\n\tHyperparameters:')
            for key, val in hyperparameters_dict.items():
                file.write(f'\n\t{key}: {val}')
            file.write(f'\n\nModel Architecture:\n{bdq_model.policy_network}')
            file.write(f'\n\n\t\tNote:\n\t\t\t{note}')

        sim.reset_state()
