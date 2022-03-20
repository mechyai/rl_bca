import os
import matplotlib.pyplot as plt
import time
import torch

import pandas as pd
import numpy as np

from torch.utils.tensorboard import SummaryWriter

# import openstudio  # ver 3.2.0 !pip list

from emspy import EmsPy, BcaEnv, MdpManager, idf_editor
from bca import Agent_TB, BranchingDQN, ReplayMemory, EpsilonGreedyStrategy, mdp

import process_results

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# -- FILE PATHS --
# E+ Download Path
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'
# IDF File / Modification Paths
repo_root = 'A:/Files/PycharmProjects/rl_bca'
os_folder = os.path.join(repo_root, 'Current_Prototype/BEM')
idf_file_name = 'IdfFiles/BEM_5z_V1_May.idf'  # ***********************************************************
idf_file_base = os.path.join(os_folder, idf_file_name)
idf_final_file = os.path.join(os_folder, 'BEM_5z_V1_May.idf')
# Weather Path
ep_weather_path = os.path.join(os_folder, 'WeatherFiles/EPW/DallasTexas_2019CST.epw')
# Output .csv Path
cvs_output_path = ''

# -- INSTANTIATE MDP --
my_mdp = MdpManager.generate_mdp_from_tc(mdp.tc_intvars, mdp.tc_vars, mdp.tc_meters, mdp.tc_weather, mdp.tc_actuators)

# -- CUSTOM SQL TRACKING --
data_tracking = {  # custom tracking for actuators, (handle + unit type)
    # -- Reward --
    'reward': ('Schedule:Constant', 'Schedule Value', 'Reward Tracker', 'Dimensionless'),
    'reward_cumulative': ('Schedule:Constant', 'Schedule Value', 'Reward Cumulative', 'Dimensionless'),
    # -- Results Metrics --
    # Comfort
    'comfort': ('Schedule:Constant', 'Schedule Value', 'Comfort Tracker', 'Dimensionless'),
    'comfort_cumulative': ('Schedule:Constant', 'Schedule Value', 'Comfort Cumulative', 'Dimensionless'),
    # RTP
    'rtp_tracker': ('Schedule:Constant', 'Schedule Value', 'RTP Tracker', 'Dimensionless'),
    'rtp_cumulative': ('Schedule:Constant', 'Schedule Value', 'RTP Cumulative', 'Dimensionless'),
    # Wind
    'wind_hvac_use': ('Schedule:Constant', 'Schedule Value', 'Wind Energy HVAC Usage Tracker', 'Dimensionless'),
    'total_hvac_use': ('Schedule:Constant', 'Schedule Value', 'Total HVAC Energy Usage Tracker', 'Dimensionless'),
    # -- Learning --
    'loss': ('Schedule:Constant', 'Schedule Value', 'Loss Tracker', 'Dimensionless'),
}
# link with ToC Actuators, remove unit types first
data_tracking_actuators = {}
for key, values in data_tracking.items():
    my_mdp.add_ems_element('actuator', key, values[0:3])  # exclude unit, leave handle

# -- Automated IDF Modification --
year = 2019
# create final file from IDF base
auto_idf_folder = os.path.join(os_folder, 'CustomIdfFiles/Automated')
idf_editor.append_idf(idf_file_base, os.path.join(auto_idf_folder, 'V1_IDF_modifications.idf'), idf_final_file)
# daylight savings & holidays
# IDFeditor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/TEXAS_CST_Daylight_Savings_{year}.idf')
# add Schedule:Files
idf_editor.append_idf(idf_final_file, os.path.join(auto_idf_folder, f'ERCOT_RTM_{year}.idf'))  # RTP
idf_editor.append_idf(idf_final_file, os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Wind.idf'))  # FMIX, wind
idf_editor.append_idf(idf_final_file, os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Solar.idf'))  # FMIX, solar
idf_editor.append_idf(idf_final_file, os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Total.idf'))  # FMIX, total
for h in range(12):  # DAM 12 hr forecast
    idf_editor.append_idf(idf_final_file,
                          os.path.join(auto_idf_folder, f'ERCOT_DAM_12hr_forecast_{year}_{h}hr_ahead.idf'))
# add Custom Meters
idf_editor.append_idf(idf_final_file, os.path.join(auto_idf_folder, 'V1_custom_meters.idf'))
# add Custom Data Tracking IDF Objs (reference ToC of Actuators)
for _, value in data_tracking.items():
    idf_editor.insert_custom_data_tracking(value[2], idf_final_file, value[3])

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)
timesteps = 60

# -- Agent Params --

# misc
action_branches = 4
interaction_ts_frequency = 15

hyperparameter_dict = {
    # --- BDQ ---
    # architecture
    'observation_dim': 40,
    'action_branches': action_branches,  # n building zones
    'action_dim': 5,
    'shared_network_size': [48],
    'value_stream_size': [24],
    'advantage_streams_size': [24],
    # hyperparameters
    'target_update_freq': 100,  #
    'learning_rate': 0.001,  #
    'gamma': 0.8,  #

    # network mods
    'td_target': 'mean',  # mean or max
    'gradient_clip_norm': 3,
    'rescale_shared_grad_factor': 1 / (1 + action_branches),

    # --- Experience Replay ---
    'replay_capacity': int(60 / ((60 / timesteps) * interaction_ts_frequency) * 24 * 14),  # 14 days
    'batch_size': 128,

    # --- Behavioral Policy ---
    'eps_start': 0.1,  # epsilon
    'eps_end': 0.05,
    'eps_decay': 0.00005,
}

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 50,
    'run_benchmark': False,
    'exploit_final_epoch': False,
    'save_model': False,
    'save_model_final_epoch': True,
    'save_results': False,
    'save_final_results': True,
    'reward_plot_title': '',
    'experiment_title': '',
    'experiment_notes': '',
    'interaction_ts_freq': interaction_ts_frequency,  # interaction ts intervals
    'load_model': r''
}

# --- Study Parameters ---
study_params = [{}]  # leave empty [{}] for No study
epoch_params = []

# ------------------------------------------------ Run Study ------------------------------------------------
folder_made = False
for i, study in enumerate(study_params):

    # -- Adjust Study Params --
    for param_name, param_value in study.items():
        hyperparameter_dict[param_name] = param_value

    # -- Create New Model Components --
    if True:
        bdq_model = BranchingDQN(
            observation_dim=hyperparameter_dict['observation_dim'],
            action_branches=hyperparameter_dict['action_branches'],  # 5 building zones
            action_dim=hyperparameter_dict['action_dim'],  # heat/cool/off
            shared_network_size=hyperparameter_dict['shared_network_size'],
            value_stream_size=hyperparameter_dict['value_stream_size'],
            advantage_streams_size=hyperparameter_dict['advantage_streams_size'],
            target_update_freq=hyperparameter_dict['target_update_freq'],
            learning_rate=hyperparameter_dict['learning_rate'],
            gamma=hyperparameter_dict['gamma'],
            td_target=hyperparameter_dict['td_target'],  # mean or max
            gradient_clip_norm=hyperparameter_dict['gradient_clip_norm'],
            rescale_shared_grad_factor=hyperparameter_dict['rescale_shared_grad_factor']
        )

        experience_replay = ReplayMemory(
            capacity=hyperparameter_dict['replay_capacity'],
            batch_size=hyperparameter_dict['batch_size']
        )

        policy = EpsilonGreedyStrategy(
            start=hyperparameter_dict['eps_start'],
            end=hyperparameter_dict['eps_end'],
            decay=hyperparameter_dict['eps_decay']
        )

    if experiment_params_dict['load_model']:
        bdq_model.import_model(experiment_params_dict['load_model'])

    for epoch in range(experiment_params_dict['epochs']):  # train under same condition

        # -- Adjust Study Params for Epoch --
        if epoch_params:
            for param_name, param_value in epoch_params[epoch].items():
                hyperparameter_dict[param_name] = param_value

        # -- Experiment Time / Naming --

        time_start = time.time()
        model_name = f'bdq_{time.strftime("%Y%m%d_%H%M")}_{epoch}.pt'

        # ---- Tensor Board ----
        TB = SummaryWriter(f'runs/full_year_{time.strftime("%m_%d_%H-%M")}_50rcap_{epoch}')

        # -- Create Building Energy Simulation Instance --
        sim = BcaEnv(ep_path=ep_path,
                     ep_idf_to_run=idf_final_file,
                     timesteps=timesteps,
                     tc_vars=my_mdp.tc_var,
                     tc_intvars=my_mdp.tc_intvar,
                     tc_meters=my_mdp.tc_meter,
                     tc_actuator=my_mdp.tc_actuator,
                     tc_weather=my_mdp.tc_weather
                     )

        # -- Instantiate RL Agent --
        my_agent = Agent_TB(emspy_sim=sim,
                            mdp=my_mdp,
                            dqn_model=bdq_model,
                            policy=policy,
                            replay_memory=experience_replay,
                            interaction_frequency=interaction_ts_frequency,
                            learning_loop=1,
                            summary_writer=TB
                            )

        # -- Benchmark -- (do once)
        if experiment_params_dict['run_benchmark']:
            learn = False
            act = False
            experiment_params_dict['run_benchmark'] = False
        else:
            learn = True
            act = True

        # -- @ Final Epoch --
        if epoch == experiment_params_dict['epochs'] - 1:
            # Save final model
            if experiment_params_dict['save_model_final_epoch']:
                experiment_params_dict['save_model'] = True
            if experiment_params_dict['save_final_results']:
                experiment_params_dict['save_results'] = True
            # Exploit final
            if experiment_params_dict['exploit_final_epoch']:
                policy.start = 0
                hyperparameter_dict['eps_start'] = 0
                policy.decay = 0
                hyperparameter_dict['eps_decay'] = 0

        # -- Set Sim Calling Point(s) & Callback Function(s) --
        sim.set_calling_point_and_callback_function(calling_point=cp,
                                                    observation_function=my_agent.observe,
                                                    actuation_function=my_agent.act_strict_setpoints,
                                                    update_state=True,
                                                    update_observation_frequency=experiment_params_dict[
                                                        'interaction_ts_freq'],
                                                    update_actuation_frequency=experiment_params_dict[
                                                        'interaction_ts_freq'],
                                                    observation_function_kwargs={
                                                        'learn': learn},  # whether or not model learns
                                                    actuation_function_kwargs={
                                                        'actuate': act}  # whether or not agent takes actions
                                                    )

        # --**-- Run Sim --**--
        sim.run_env(ep_weather_path)
        sim.reset_state()

        # -- RECORD RESULTS --
        TB.add_scalar('__Epoch/Total Loss', my_agent.loss_total, epoch)
        TB.add_scalar('__Epoch/Reward/All Reward', my_agent.reward_sum, epoch)
        TB.add_scalar('__Epoch/Reward/Comfort Reward', my_agent.reward_component_sum[0], epoch)
        TB.add_scalar('__Epoch/Reward/RTP-HVAC Reward', my_agent.reward_component_sum[1], epoch)
        TB.add_scalar('__Epoch/Reward/Wind-HVAC Reward', my_agent.reward_component_sum[2], epoch)
        # Sim Results
        TB.add_scalar('__Epoch/_Results/Comfort Dissatisfied Total', my_agent.comfort_dissatisfaction_total, epoch)
        TB.add_scalar('__Epoch/_Results/HVAC RTP Cost Total', my_agent.hvac_rtp_costs_total, epoch)

        # -- Get Sim DFs --
        dfs = sim.get_df()

        # -- Save / Write Data --
        if experiment_params_dict['save_model'] or experiment_params_dict['save_results']:

            if not folder_made:
                # create folder for experiment, Once
                folder_made = True
                experiment_time = time.strftime("%Y%m%d_%H%M")
                experiment_name = f'Exp_{experiment_time}'
                folder = os.path.join('Output_Saved', 'Tuning_Data', experiment_name)
                results_file_path = os.path.join(folder, f'bdq_report_{experiment_time}.txt')
                os.mkdir(folder)

            # save model (don't save benchmark model if only 1 epoch)
            if experiment_params_dict['save_model'] and not (epoch == 1 and experiment_params_dict['run_benchmark']):
                torch.save(bdq_model.policy_network.state_dict(), os.path.join(folder, model_name))  # save model

            # save results
            if experiment_params_dict['save_results']:
                with open(results_file_path, 'a+') as file:
                    file.write(f'\n\n Experiment Descp: {experiment_params_dict["experiment_title"]}')
                    file.write(f'\n\n Model Name: {model_name}')
                    file.write(f'\n\tTime Train = {round(time_start - time.time(), 2) / 60} mins')
                    file.write(f'\n\t*Epochs trained = {epoch}')
                    file.write(f'\n\t******* Cumulative Reward = {my_agent.reward_sum}')
                    file.write(f'\n\t*Performance Metrics:')
                    file.write(f'\n\t\tDiscomfort Metric = {my_agent.comfort_dissatisfaction_total}')
                    file.write(f'\n\t\tRTP HVAC Cost Metric = {my_agent.hvac_rtp_costs_total}')
                    file.write(f'\n\n\tState Space: {my_agent.state_var_names}')
                    file.write('\n\n\tHyperparameters:')
                    for key, val in hyperparameter_dict.items():
                        file.write(f'\n\t\t{key}: {val}')
                    file.write(f'\n\nModel Architecture:\n{bdq_model.policy_network}')
                    file.write(f'\n\n\t\tNotes:\n\t\t\t{experiment_params_dict["experiment_notes"]}')
