import os
import matplotlib.pyplot as plt
import time
import torch

import pandas as pd
import numpy as np

# import openstudio  # ver 3.2.0 !pip list

from emspy import EmsPy, BcaEnv, MdpManager, idf_editor
from bca import Agent, BranchingDQN, ReplayMemory, EpsilonGreedyStrategy, mdp_manager

import process_results

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

# -- FILE PATHS --
# E+ Download Path
ep_path = 'A:/Programs/EnergyPlusV9-5-0/'
# IDF File / Modification Paths
repo_root = 'A:/Files/PycharmProjects/rl_bca'
os_folder = os.path.join(repo_root, 'Current_Prototype/BEM')
idf_file_name = 'IdfFiles/BEM_V1_2019_Year.idf'  # ***********************************************************
idf_file_base = os.path.join(os_folder, idf_file_name)
idf_final_file = os.path.join(os_folder, 'BEM_V1_2019_Year.idf')
# Weather Path
ep_weather_path = os.path.join(os_folder, 'WeatherFiles/EPW/DallasTexas_2019CST.epw')
# Output .csv Path
cvs_output_path = ''


# -- INSTANTIATE MDP --
my_mdp = MdpManager.generate_mdp_from_tc(mdp_manager.tc_intvars, mdp_manager.tc_vars, mdp_manager.tc_meters, mdp_manager.tc_weather, mdp_manager.tc_actuators)


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
interaction_ts_frequency = 10

hyperparameter_dict = {
    # --- BDQ ---
    # architecture
    'observation_dim': 40,
    'action_branches': action_branches,  # n building zones
    'action_dim': 5,
    'shared_network_size': [96, 96],
    'value_stream_size': [48],
    'advantage_streams_size': [56],
    # hyperparameters
    'target_update_freq': 250,  #
    'learning_rate': 0.0005,  #
    'gamma': 0.5,  #

    # network mods
    'td_target': 'mean',  # mean or max
    'gradient_clip_norm': 5,
    'rescale_shared_grad_factor': 1 / (1 + action_branches),

    # --- Experience Replay ---
    'replay_capacity': int(60 / ((60 / timesteps) * interaction_ts_frequency) * 24 * 14),  # 14 days
    'batch_size': 128,

    # --- Behavioral Policy ---
    'eps_start': 0.15,  # epsilon
    'eps_end': 0.1,
    'eps_decay': 0.00005,
}

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 20,
    'run_benchmark': True,
    'exploit_final_epoch': False,
    'save_model': True,
    'save_model_final_epoch': True,
    'save_results': True,
    'save_final_results': True,
    'reward_plot_title': '',
    'experiment_title': '',
    'experiment_notes': '',
    'interaction_ts_freq': interaction_ts_frequency,  # interaction ts intervals
    'load_model': r''
}

# --- Study Parameters ---
study_params = [{}]  # , {'gamma': 0.8}, {'gamma': 0.9}]  # leave empty [{}] for No study
epoch_params = []  # 'learning_rate': 0.01}, {'learning_rate': 0.005}, {'learning_rate': 0.001},
#                 {'learning_rate': 0.001}, {'learning_rate': 0.001}, {'learning_rate': 0.0005},
#                 {'learning_rate': 0.0005}, {'learning_rate': 0.0005}, {'learning_rate': 0.0005},
#                 {'learning_rate': 0.0001}, {'learning_rate': 0.00005}, {'learning_rate': 0.0001}]

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

        # -- Clear Up Memory --
        if 'dfs' in locals():
            del dfs, df_loss, df_energy_source
            del rtp_hvac_agent, rtp_hvac_control, rtp_all

        time_start = time.time()
        model_name = f'bdq_{time.strftime("%Y%m%d_%H%M")}_{epoch}.pt'

        # -- Adjust Study Params for Epoch --
        if epoch_params:
            for param_name, param_value in epoch_params[epoch].items():
                hyperparameter_dict[param_name] = param_value

        # -- Create Building Energy Simulation Instance --
        sim = BcaEnv(ep_path, idf_final_file, timesteps,
                     my_mdp.tc_var, my_mdp.tc_intvar, my_mdp.tc_meter,
                     my_mdp.tc_actuator, my_mdp.tc_weather)

        # -- Instantiate RL Agent --
        my_agent = Agent(sim, my_mdp, bdq_model, policy, experience_replay, interaction_ts_frequency, learning_loop=1)

        # Do once
        if experiment_params_dict['run_benchmark']:
            learn = False
            act = False
            experiment_params_dict['run_benchmark'] = False  # TODO why does benchmark take so long without learning
        else:
            learn = True
            act = True

        # -- @ Final Epoch --
        if epoch == experiment_params_dict['epochs'] - 1:
            # save final model
            if experiment_params_dict['save_model_final_epoch']:
                experiment_params_dict['save_model'] = True
            if experiment_params_dict['save_final_results']:
                experiment_params_dict['save_results'] = True
            # exploit final
            if experiment_params_dict['exploit_final_epoch']:
                learning = False
                policy.start = 0
                hyperparameter_dict['eps_start'] = 0
                policy.decay = 0
                hyperparameter_dict['eps_decay'] = 0

        # -- Set Sim Calling Point(s) & Callback Function(s) --
        sim.set_calling_point_and_callback_function(cp, my_agent.observe, my_agent.act_strict_setpoints, True,
                                                    experiment_params_dict['interaction_ts_freq'],
                                                    experiment_params_dict['interaction_ts_freq'],
                                                    observation_function_kwargs={
                                                        'learn': learn
                                                    },
                                                    actuation_function_kwargs={
                                                        'actuate': act
                                                    })

        # ------------------------------------------------ Run Sim ------------------------------------------------
        sim.run_env(ep_weather_path)
        sim.reset_state()

        # -- Get Sim DFs --
        ts_mod = 0
        # Reward
        dfs = sim.get_df()
        dfs['reward']['cumulative'] = dfs['reward'][['reward']].cumsum()  # create cumulative reward column
        cumulative_reward = float(dfs['reward'][['cumulative']].iloc[-1])  # get final cumulative reward
        # Loss
        # get at interaction interval
        df_loss = dfs['actuator'][['Datetime', 'Timestep', 'loss']]
        df_loss = df_loss[df_loss['Timestep'] % experiment_params_dict['interaction_ts_freq'] == ts_mod]

        # RTP
        rtp_hvac_agent = my_agent.rtp_histogram_data
        rtp_hvac_control = np.load(
            r'A:\Files\PycharmProjects\rl_bca\Current_Prototype\Output_Saved\rtp_histo_control.npy')
        n_controlled_zones = action_branches
        rtp_all = np.tile(dfs['var'][['rtp']].to_numpy().squeeze(1), n_controlled_zones)

        # Wind
        df_energy_source = dfs['actuator'][['Datetime', 'Timestep', 'wind_hvac_use', 'total_hvac_use']]
        df_energy_source = df_energy_source.set_index(
            pd.DatetimeIndex(df_energy_source['Datetime']))  # make datetime index

        monthly_wind_hvac_usage_proportion = []
        months_int = range(1, 13, 1)
        for month in months_int:
            energy_month = df_energy_source[df_energy_source.index.month == month]  # get data for single month
            energy_month_ts = energy_month[
                energy_month['Timestep'] % experiment_params_dict['interaction_ts_freq'] == ts_mod
                ]
            monthly_wind_use = energy_month['wind_hvac_use'].sum() / energy_month['total_hvac_use'].sum()
            monthly_wind_hvac_usage_proportion.append(monthly_wind_use)

        # -- Plot Results --
        # Reward
        fig, axs = plt.subplots(nrows=2, ncols=3)
        plt.gcf()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()

        axs[0, 0].plot(dfs['reward'][['Datetime']], dfs['reward'][['reward']])
        axs0_2 = axs[0, 0].twinx()
        axs0_2.plot(dfs['reward'][['Datetime']], dfs['reward'][['cumulative']])
        axs[0, 0].set_title('Reward')
        axs0_2.set_ylabel('Cumulative')
        axs[0, 0].set_ylabel('Discrete')
        axs[0, 0].grid()
        # Loss
        axs[1, 0].set_yscale('log')
        axs[1, 0].plot(df_loss['Datetime'], df_loss['loss'])
        axs[1, 0].set_title('Log Loss')
        axs[1, 0].grid()
        # RTP
        # plt.yscale('log', nonposy='clip')
        hist_range = (0, 50)
        n_bins = 50
        axs[0, 1].hist(rtp_all, bins=n_bins, range=hist_range, alpha=0.75, label='All Time', color='k')
        axs[0, 1].hist(rtp_hvac_agent, bins=n_bins, range=hist_range, alpha=0.5, label="Agent")
        axs[0, 1].hist(rtp_hvac_control, bins=n_bins, range=hist_range, alpha=0.75, label="Control")
        axs[0, 1].set_title('RTP Price Awareness')
        axs[0, 1].set_xlabel('$RTP Bins - Lower Range')
        axs[0, 1].set_ylabel('RTP HVAC Usage')
        axs[0, 1].grid()
        axs[0, 1].legend()
        # higher scale
        hist_range = (500, 10000)
        n_bins = 50
        axs[0, 1].hist(rtp_all, bins=n_bins, range=hist_range, alpha=0.75, label='All Time', color='k')
        axs[0, 1].hist(rtp_hvac_agent, bins=n_bins, range=hist_range, alpha=0.5, label="Agent")
        axs[0, 1].hist(rtp_hvac_control, bins=n_bins, range=hist_range, alpha=0.75, label="Control")
        axs[1, 1].set_xlabel('$RTP Bins - Lower Range')
        axs[1, 1].set_ylabel('RTP HVAC Usage')
        axs[1, 1].set_ylim([0, 1])
        axs[1, 1].grid()

        # Wind Use
        axs[0, 2].bar(months_int, monthly_wind_hvac_usage_proportion)
        axs[0, 2].set_title('HVAC Renewable Energy Usage')
        axs[0, 2].set_xlabel('Months')
        axs[0, 2].set_ylabel('% HVAC Energy from Wind')
        axs[0, 2].set_ylim([0, 1])
        axs[0, 2].grid()

        plt.tight_layout()

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

            if experiment_params_dict['save_final_results'] and epoch == experiment_params_dict['epochs'] - 1:
                # save RTP data
                np.save(os.path.join(folder, experiment_time + '_rtp_histo_agent'),
                        np.asarray(my_agent.rtp_histogram_data))

            # save results
            if experiment_params_dict['save_results']:
                plot_results_name = model_name[:-3] + '_results.png'
                fig.savefig(os.path.join(folder, plot_results_name), bbox_inches='tight')
                plt.close('all')

                with open(results_file_path, 'a+') as file:
                    file.write(f'\n\n Experiment Descp: {experiment_params_dict["experiment_title"]}')
                    file.write(f'\n\n Model Name: {model_name}')
                    file.write(f'\nReward Plot Name: {plot_results_name}')
                    file.write(f'\n\tTime Train = {round(time_start - time.time(), 2) / 60} mins')
                    file.write(f'\n\t*Epochs trained = {epoch}')
                    file.write(f'\n\t******* Cumulative Reward = {cumulative_reward}')
                    file.write(f'\n\t*Performance Metrics:')
                    file.write(f'\n\t\tDiscomfort Metric = {my_agent.comfort_dissatisfaction_total}')
                    file.write(f'\n\t\tRTP HVAC Cost Metric = {my_agent.hvac_rtp_costs_total}')
                    file.write(f'\n\n\tState Space: {my_agent.state_var_names}')
                    file.write('\n\n\tHyperparameters:')
                    for key, val in hyperparameter_dict.items():
                        file.write(f'\n\t\t{key}: {val}')
                    file.write(f'\n\nModel Architecture:\n{bdq_model.policy_network}')
                    file.write(f'\n\n\t\tNotes:\n\t\t\t{experiment_params_dict["experiment_notes"]}')
