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
from emspy import emspy
from emspy import idf_editor

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

# --- create EMS Table of Contents (TC) for sensors/actuators ---

# vars_tc = {"attr_handle_name": ["variable_type", "variable_key"],...}
# int_vars_tc = {"attr_handle_name": "variable_type", "variable_key"],...}
# meters_tc = {"attr_handle_name": "meter_name",...}
# actuators_tc = {"attr_handle_name": ["component_type", "control_type", "actuator_key"],...}
# weather_tc = {"attr_name": "weather_metric",...}

# STATE SPACE (& Auxiliary Simulation Data)
zn0 = 'Core_ZN ZN'
zn1 = 'Perimeter_ZN_1 ZN'
zn2 = 'Perimeter_ZN_2 ZN'
zn3 = 'Perimeter_ZN_3 ZN'
zn4 = 'Perimeter_ZN_4 ZN'

intvars_tc = {}

tc_meters = {
    # Building-wide
    'electricity_facility': 'Electricity:Facility',
    'electricity_HVAC': 'Electricity:HVAC',
    'electricity_heating': 'Heating:Electricity',
    'electricity_cooling': 'Cooling:Electricity',
    'gas_heating': 'NaturalGas:HVAC',
    # Solar (custom)
    'PV_generation_meter': 'Solar Generation',
    # Zn0 (custom meters)
    'zn0_heating_electricity': 'Zn0 HVAC Heating Electricity',
    'zn0_heating_gas': 'Zn0 HVAC Heating Natural Gas',
    'zn0_cooling_electricity': 'Zn0 HVAC Cooling Electricity',
    # Zn1 (custom meters)
    'zn1_heating_electricity': 'Zn1 HVAC Heating Electricity',
    'zn1_heating_gas': 'Zn1 HVAC Heating Natural Gas',
    'zn1_cooling_electricity': 'Zn1 HVAC Cooling Electricity',
    # Zn2 (custom meters)
    'zn2_heating_electricity': 'Zn2 HVAC Heating Electricity',
    'zn2_heating_gas': 'Zn2 HVAC Heating Natural Gas',
    'zn2_cooling_electricity': 'Zn2 HVAC Cooling Electricity',
    # Zn3 (custom meters)
    'zn3_heating_electricity': 'Zn3 HVAC Heating Electricity',
    'zn3_heating_gas': 'Zn3 HVAC Heating Natural Gas',
    'zn3_cooling_electricity': 'Zn3 HVAC Cooling Electricity',
    # Zn4 (custom meters)
    'zn4_heating_electricity': 'Zn4 HVAC Heating Electricity',
    'zn4_heating_gas': 'Zn4 HVAC Heating Natural Gas',
    'zn4_cooling_electricity': 'Zn4 HVAC Cooling Electricity',
}

tc_vars = {
    # Building
    'building_hrs': ['Schedule Value', 'OfficeSmall HVACOperationSchd'],  # is building 'open'/'close'?
    'PV_generation': ['Generator Produced DC Electricity Energy', 'Generator Photovoltaic 1'],  # [J] HVAC, Sum
    # Zone 0
    'zn0_temp': ['Zone Air Temperature', zn0],
    'zn0_RH': ['Zone Air Relative Humidity', zn0],
    'zn0_ppl': ['Zone People Occupant Count', zn0],
    # Zone 1
    'zn1_temp': ['Zone Air Temperature', zn1],
    'zn1_RH': ['Zone Air Relative Humidity', zn1],
    'zn1_ppl': ['Zone People Occupant Count', zn1],
    # Zone 2
    'zn2_temp': ['Zone Air Temperature', zn2],
    'zn2_RH': ['Zone Air Relative Humidity', zn2],
    'zn2_ppl': ['Zone People Occupant Count', zn2],
    # Zone 3
    'zn3_temp': ['Zone Air Temperature', zn3],
    'zn3_RH': ['Zone Air Relative Humidity', zn3],
    'zn3_ppl': ['Zone People Occupant Count', zn3],
    # Zone 4
    'zn4_temp': ['Zone Air Temperature', zn4],
    'zn4_RH': ['Zone Air Relative Humidity', zn4],
    'zn4_ppl': ['Zone People Occupant Count', zn4]
}

tc_weather = {  # used for current and forecasted weather
    'oa_rh': 'outdoor_relative_humidity',
    'oa_db': 'outdoor_dry_bulb',
    'oa_pa': 'outdoor_barometric_pressure',
    'sun_up': 'sun_is_up',
    'rain': 'is_raining',
    'snow': 'is_snowing',
    'wind_dir': 'wind_direction',
    'wind_speed': 'wind_speed'
}

# ACTION SPACE (& Auxiliary Control)
tc_actuators = {
    # -- CONTROL --
    # HVAC Control Setpoints
    'zn0_cooling_sp': ['Zone Temperature Control', 'Cooling Setpoint', zn0],
    'zn0_heating_sp': ['Zone Temperature Control', 'Heating Setpoint', zn0],
    'zn1_cooling_sp': ['Zone Temperature Control', 'Cooling Setpoint', zn1],
    'zn1_heating_sp': ['Zone Temperature Control', 'Heating Setpoint', zn1],
    'zn2_cooling_sp': ['Zone Temperature Control', 'Cooling Setpoint', zn2],
    'zn2_heating_sp': ['Zone Temperature Control', 'Heating Setpoint', zn2],
    'zn3_cooling_sp': ['Zone Temperature Control', 'Cooling Setpoint', zn3],
    'zn3_heating_sp': ['Zone Temperature Control', 'Heating Setpoint', zn3],
    'zn4_cooling_sp': ['Zone Temperature Control', 'Cooling Setpoint', zn4],
    'zn4_heating_sp': ['Zone Temperature Control', 'Heating Setpoint', zn4],
}

# -- CUSTOM TRACKING --
reward_unit_type = "Dimensionless"
pv_unit_type = "Dimensionless"
data_tracking = {  # custom tracking
    'reward': ['Schedule:Constant', 'Schedule Value', 'Reward Tracker', reward_unit_type],
    'reward_cumulative': ['Schedule:Constant', 'Schedule Value', 'Reward Cumulative', reward_unit_type],
    'net_facility_pv': ['Schedule:Constant', 'Schedule Value', 'Facility Electricity and PV Generation Difference', pv_unit_type],
    'net_hvac_pv': ['Schedule:Constant', 'Schedule Value', 'HVAC Electricity and PV Generation Difference', pv_unit_type]
}
# link with ToC Actuators, remove unit types first
data_tracking_actuators = {}
for key, values in data_tracking.items():
    data_tracking_actuators[key] = values[0:3]  # remove unit type
tc_actuators.update(data_tracking_actuators)  # input to Actuator ToC

# Automated IDF Modification
# add misc, schedules sensors, etc.
idf_editor.append_idf(idf_file_base, r'BEM/CustomIdfFiles/Automated/V1_IDF_modifications.idf', idf_final_file)
# add Custom Meters
idf_editor.append_idf(idf_file_base, r'BEM/CustomIdfFiles/Automated/V1_custom_meters.idf', idf_final_file)
# add Custom Data Tracking IDF Objs (reference ToC of Actuators)
for _, value in data_tracking.items():
    idf_editor.insert_custom_data_tracking(value[2], idf_final_file, value[3])

# Simulation Params
cp = emspy.EmsPy.available_calling_points[6]  # 5-15 valid for timestep loop
timesteps = 4

# Create Building Energy Simulation Instance
sim = emspy.BcaEnv(ep_path, idf_final_file, timesteps, tc_vars, {}, tc_meters, tc_actuators, tc_weather)


class Agent:
    def __init__(self):
        self.net_facility_pv = 0
        self.net_hvac_pv = 0

    def observe(self):
        # ems & weather
        meters = sim.get_ems_data('meter')
        vars = sim.get_ems_data('var')
        weather = sim.get_ems_data('weather')
        # meters
        facility_elec_meter = sim.get_ems_data('electricity_facility')
        hvac_elec_meter = sim.get_ems_data('electricity_HVAC')
        pv_gen_meter = sim.get_ems_data('PV_generation_meter')
        self.net_facility_pv = facility_elec_meter - pv_gen_meter
        self.net_hvac_pv = hvac_elec_meter - pv_gen_meter

    def act(self):

        return {
            # Actuation
            'zn0_heating_sp': 25, 'zn0_cooling_sp': 29,
            'zn1_heating_sp': 25, 'zn1_cooling_sp': 29,
            'zn2_heating_sp': 25, 'zn2_cooling_sp': 29,
            'zn3_heating_sp': 25, 'zn3_cooling_sp': 29,
            'zn4_heating_sp': 25, 'zn4_cooling_sp': 29,
            # Data Tracking
            'reward': 0,
            'reward_cumulative': 0,
            'net_facility_pv': self.net_facility_pv,
            'net_hvac_pv': self.net_hvac_pv,
        }

    def c_to_f(self, temp_c: float):
        return 1.8 * temp_c + 32


# Instantiate RL Agent
agent = Agent()

# Set Calling Point(s) & Callback Function(s)
sim.set_calling_point_and_callback_function(cp, agent.observe, agent.act, True, 1, 1)

# Init Custom DFs
sim.init_custom_dataframe_dict('PV Tracker', cp, 1, ['PV_generation_meter','PV_generation'])

# RUN Simulation
sim.run_env(ep_weather_path)
sim.reset_state()
# GET SIM DFs
dfs = sim.get_df()
