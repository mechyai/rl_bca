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
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# import openstudio  # ver 3.2.0 !pip list
from emspy import emspy, idf_editor, mdpmanager
from Current_Prototype import MDP_v1 as MDP

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# FILE PATHS
# E+ Download Path
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'
# IDF File / Modification Paths
idf_file_name = r'/BEM_5z_Prototype_PV_V2/run/BEM_V1_2019_Year.idf'
idf_final_file = r'A:/Files/PycharmProjects/RL-BCA/Current_Prototype/BEM/BEM_5z_V1.idf'
os_folder = r'A:/Files/PycharmProjects/RL-bca/Current_Prototype/BEM'
idf_file_base = os_folder + idf_file_name
# Weather Path
ep_weather_path = os_folder + r'/WeatherFiles/EPW/DallasTexas_2019CST.epw'
# Output .csv Path
cvs_output_path = ''

# establish MDP
mdp = MDP.main()

# -- CUSTOM TRACKING --
data_tracking = {  # custom tracking (handle + unit type)
    'reward': ('Schedule:Constant', 'Schedule Value', 'Reward Tracker', 'Dimensionless'),
    'reward_cumulative': ('Schedule:Constant', 'Schedule Value', 'Reward Cumulative', 'Dimensionless'),
    'wind_gen_relative': ('Schedule:Constant', 'Schedule Value', 'Wind Gen of Total', 'Dimensionless')
}
# link with ToC Actuators, remove unit types first
data_tracking_actuators = {}
for key, values in data_tracking.items():
    mdp.add_ems_element('actuator', key, values[0:3])  # exclude unit, leave handle

# Automated IDF Modification
year = 2019
# create final file from IDF base
idf_editor.append_idf(idf_file_base, r'BEM/CustomIdfFiles/Automated/V1_IDF_modifications.idf', idf_final_file)
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

# Simulation Params
cp = emspy.EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop
timesteps = 4

# Create Building Energy Simulation Instance
sim = emspy.BcaEnv(ep_path, idf_final_file, timesteps,
                   mdp.tc_var, mdp.tc_intvar, mdp.tc_meter, mdp.tc_actuator, mdp.tc_weather)


class Agent:
    def __init__(self):
        # FREQUENCY - INTERACTION
        self.observation_ts = 1  # how often agent will observe state & keep fixed action - off-policy
        self.action_ts = 15  # how often agent will observe state & act - on-policy
        self.action_delay = 5  # how many ts will agent be fixed at beginning of simulation


        # TODO figure out fundamental vs. derived state space params
        # State Space
        # from EMS vars
        self.vars = list(mdp.tc_var.keys())
        self.meters = list(mdp.tc_meter.keys())
        self.weather = list(mdp.tc_weather.keys())

        self.wind_gen_relative = 0

    def observe(self):

        # FETCH/UPDATE EMS DATA
        vars = mdp.update_ems_value(self.vars, sim.get_ems_data(self.vars))
        meters = mdp.update_ems_value(self.meters, sim.get_ems_data(self.meters))
        weather = mdp.update_ems_value(self.weather, sim.get_ems_data(self.weather))
        #
        # Manual Encoding
        mdp.wind_gen_encoding_fxn_args[1] = mdp.read_ems_values(['total_gen'])  # change max val for normalization
        # UPDATE w/ NEW ENCODING
        vars = mdp.get_ems_encoded_values(self.vars)
        # self.wind_gen_relative = vars[self.vars.index('wind_gen')]

        print(f'\n\nVars: {vars}\nMeters: {meters}\nWeather: {weather}')

    def act(self):

        return {
            # Actuation
            'zn0_heating_sp': 25, 'zn0_cooling_sp': 29,
            'zn1_heating_sp': 25, 'zn1_cooling_sp': 29,
            'zn2_heating_sp': 25, 'zn2_cooling_sp': 29,
            'zn3_heating_sp': 25, 'zn3_cooling_sp': 29,
            'zn4_heating_sp': 25, 'zn4_cooling_sp': 29,
            # Data Tracking
            # 'reward': 0,
            # 'reward_cumulative': 0,
            'wind_gen_relative': self.wind_gen_relative
        }

    def c_to_f(self, temp_c: float):
        return 1.8 * temp_c + 32


# Instantiate RL Agent
agent = Agent()

# Set Calling Point(s) & Callback Function(s)
sim.set_calling_point_and_callback_function(cp, agent.observe, agent.act, True, 1, 1)

# RUN Simulation
sim.run_env(ep_weather_path)
sim.reset_state()
# GET SIM DFs
dfs = sim.get_df()
