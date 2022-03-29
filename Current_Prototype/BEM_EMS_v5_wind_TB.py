import os
import gc
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter

from emspy import EmsPy
from bca import ModelManager, RunManager, mdp_manager, paths_config

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(paths_config.repo_root, 'Current_Prototype/BEM')
idf_file_base = os.path.join(bem_folder, 'IdfFiles/BEM_5z_V1_baseline_May.idf')  # !--------------------------------------------
idf_final_file = os.path.join(bem_folder, 'BEM_5z_V1.idf')
# Weather Path
epw_file = os.path.join(bem_folder, 'WeatherFiles/EPW/DallasTexas_2019CST.epw')

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 5,
    'run_benchmark': True,
    'exploit_final_epoch': False,
    'save_model': True,
    'save_model_final_epoch': True,
    'save_results': False,
    'save_final_results': True,
    'reward_plot_title': '',
    'experiment_title': '',
    'experiment_notes': '',
    'load_model': r''
}

# -- INSTANTIATE MDP / CUSTOM IDF / CREATE SIM --
my_model = ModelManager(
    mdp_manager_file=mdp_manager,
    idf_file_input=idf_file_base,
    idf_file_output=idf_final_file,
    year=2019
)
my_model.create_custom_idf()

# --- Study Parameters ---
run_manager = RunManager()
# runs = run_manager.shuffle_runs()
runs = run_manager.get_runs(run_manager.selected_params)

# ------------------------------------------------ Run Study ------------------------------------------------
runs_limit = 1
for i, run in enumerate(runs):

    # -- Create New Model Components --
    my_bdq = run_manager.create_bdq(run, rnn=True)

    # Load model, if desired
    if experiment_params_dict['load_model']:
        my_bdq.import_model(experiment_params_dict['load_model'])

    # -- Benchmark -- (do once)
    if experiment_params_dict['run_benchmark']:
        learn = False
        act = False
        experiment_params_dict['run_benchmark'] = False
    else:
        learn = True
        act = True

    for epoch in range(experiment_params_dict['epochs']):  # train under same condition

        # ---- Tensor Board ----
        TB = SummaryWriter(comment=f'_run{i}_epoch{epoch}-{experiment_params_dict["epochs"]}')

        if 'my_mdp' in locals():
            del my_mdp, my_sim, my_policy, my_memory, my_agent
            gc.collect()  # release memory

        # -- Create MDP & Building Sim Instance --
        my_mdp = my_model.create_mdp()
        my_sim = my_model.create_sim(my_mdp)

        # -- Instantiate RL Agent --
        my_policy = run_manager.create_policy(run)
        my_memory = run_manager.create_exp_replay(run, rnn=True)
        my_agent = run_manager.create_agent(run, my_mdp, my_sim, TB)

        # -- Set Sim Calling Point(s) & Callback Function(s) --
        my_sim.set_calling_point_and_callback_function(
            calling_point=cp,
            observation_function=my_agent.observe,
            actuation_function=my_agent.act_strict_setpoints,
            update_state=True,
            update_observation_frequency=run.interaction_ts_frequency,
            update_actuation_frequency=run.interaction_ts_frequency,
            observation_function_kwargs={'learn': learn},  # whether or not model learns
            actuation_function_kwargs={'actuate': act}  # whether or not agent takes actions
        )

        # --**-- Run Sim --**--
        my_sim.run_env(epw_file)
        my_sim.reset_state()

        # -- RECORD RESULTS --
        print(run)
        TB.add_scalar('__Epoch/Total Loss', my_agent.loss_total, epoch)
        TB.add_scalar('__Epoch/Reward/All Reward', my_agent.reward_sum, epoch)
        TB.add_scalar('__Epoch/Reward/Comfort Reward', my_agent.reward_component_sum[0], epoch)
        TB.add_scalar('__Epoch/Reward/RTP-HVAC Reward', my_agent.reward_component_sum[1], epoch)
        TB.add_scalar('__Epoch/Reward/Wind-HVAC Reward', my_agent.reward_component_sum[2], epoch)
        # Sim Results
        TB.add_scalar('__Epoch/_Results/Comfort Dissatisfied Total', my_agent.comfort_dissatisfaction_total, epoch)
        TB.add_scalar('__Epoch/_Results/HVAC RTP Cost Total', my_agent.hvac_rtp_costs_total, epoch)
        # Hyperparameter
        TB.add_hparams(
            hparam_dict=
            {
                **{
                    'run': i,
                    'epoch': epoch
                   },
                **run._asdict()
            },
            metric_dict=
            {
                'Total Reward': my_agent.reward_sum,
                'Total Loss': my_agent.loss_total,
                'Comfort Reward': my_agent.reward_component_sum[0],
                'RTP-HVAC Reward': my_agent.reward_component_sum[1],
                'Wind-HVAC Reward': my_agent.reward_component_sum[2],
                'Comfort Dissatisfied Metric': my_agent.comfort_dissatisfaction_total,
                'HVAC RTP Cost Metric': my_agent.hvac_rtp_costs_total
            },
            hparam_domain_discrete=
            {
                **{
                    'run': list(range(runs_limit)),
                    'epoch': list(range(experiment_params_dict['epochs']))
                },
                **RunManager.hyperparameter_dict
            },
        )

        # for name, param in my_bdq.policy_network.named_parameters():
        #     TB_1.add_histogram(name, param, epoch)
        #     TB.add_histogram(f'{name}.grad', param.grad, epoch)

        if epoch % 5 == 0:
            torch.save(my_bdq.policy_network.state_dict(), os.path.join(f'lstm_50_epoch{epoch}_model'))  # save model

        # Only need 1 baseline epoch
        if not act:
            continue

    # -- Save Model (don't save benchmark model if only 1 epoch)
    if experiment_params_dict['save_model'] and not (epoch == 1 and experiment_params_dict['run_benchmark']):
        torch.save(my_bdq.policy_network.state_dict(), os.path.join('lstm_50_epoch_model'))  # save model

    if i >= runs_limit - 1:
        "Breaking from loop"
        break
