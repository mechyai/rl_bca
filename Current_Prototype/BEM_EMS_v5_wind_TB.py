import os
import gc
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from emspy import EmsPy
from bca import ModelManager, RunManager, TensorboardManager, mdp_manager, _paths_config

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
idf_file_base = os.path.join(bem_folder, 'IdfFiles/BEM_V1_2019_baseline_May.idf')  # !-----------------------------------
idf_final_file = os.path.join(bem_folder, 'BEM_V1_2019_Year.idf')
# Weather Path
epw_file = os.path.join(bem_folder, 'WeatherFiles/EPW/DallasTexas_2019CST.epw')

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 1,
    'run_benchmark': True,
    'exploit_final_epoch': False,
    'save_model': True,
    'save_model_final_epoch': True,
    'save_results': False,
    'save_final_results': True,
    'reward_plot_title': '',
    'experiment_title': '',
    'experiment_notes': '',
    'load_model': r'A:\Files\PycharmProjects\rl_bca\Current_Prototype\MEAN_reward_epoch_33_lr_1e-06'
}

# -- INSTANTIATE MDP / CUSTOM IDF / CREATE SIM --
my_model = ModelManager(
    mdp_manager_file=mdp_manager,
    idf_file_input=idf_file_base,
    idf_file_output=idf_final_file,
    year=mdp_manager.year
)
my_model.create_custom_idf()

# --- Study Parameters ---
run_manager = RunManager()
# runs = run_manager.shuffle_runs()
runs = [run_manager.selected_params]

# ------------------------------------------------ Run Study ------------------------------------------------
runs_limit = 1
for i, run in enumerate(runs):

    # -- Create New Model Components --
    my_bdq = run_manager.create_bdq(run, rnn=run.rnn)

    # Load model, if desired
    if experiment_params_dict['load_model']:
        my_bdq.import_model(experiment_params_dict['load_model'])
    j = 4

    for epoch in range(experiment_params_dict['epochs']):  # train under same condition

        # -- Benchmark -- (do once)
        if experiment_params_dict['run_benchmark']:
            learn = False
            act = False
            exploit = False
            experiment_params_dict['run_benchmark'] = False
        else:
            if experiment_params_dict['exploit_final_epoch'] and epoch == (experiment_params_dict['epochs'] - 1):
                exploit = True
                learn = False
                act = True
            else:
                exploit = False
                learn = True
                act = True

        if epoch % 5 == 0:
            lr = [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6][j]
            print(f'Learning rate: {lr}')
            j += 1
            run = run._replace(learning_rate=lr)
            my_bdq.change_learning_rate_discrete(lr)

        # ---- Tensor Board ----
        TB = TensorboardManager(run_manager, comment=f'__MEAN_cont_14_lr_{lr}_epoch{epoch + 1}-{experiment_params_dict["epochs"]}')
        # TB = TensorboardManager(run_manager, comment=f'__May_epoch_35_EXPLOIT')
        # TB = TensorboardManager(run_manager, comment=f'___May_BASELINE')

        if 'my_mdp' in locals():
            del my_mdp, my_sim, my_policy, my_memory, my_agent
            gc.collect()  # release memory

        # -- Create MDP & Building Sim Instance --
        my_mdp = my_model.create_mdp()
        my_sim = my_model.create_sim(my_mdp)

        # -- Instantiate RL Agent --
        my_policy = run_manager.create_policy(run)
        my_memory = run_manager.create_exp_replay(run, rnn=run.rnn)
        my_agent = run_manager.create_agent(run, my_mdp, my_sim, TB)

        # -- Set Sim Calling Point(s) & Callback Function(s) --
        my_sim.set_calling_point_and_callback_function(
            calling_point=cp,
            observation_function=my_agent.observe,
            actuation_function=my_agent.act_step_strict_setpoints,  # Try different actuation functions
            update_state=True,
            update_observation_frequency=run.interaction_ts_frequency,
            update_actuation_frequency=run.interaction_ts_frequency,
            observation_function_kwargs={'learn': learn},
            actuation_function_kwargs={'actuate': act, 'exploit': exploit}
        )

        # --**-- Run Sim --**--
        my_sim.run_env(epw_file)
        my_sim.reset_state()

        # -- RECORD RESULTS --
        print(run)
        TB.record_epoch_results(my_agent, experiment_params_dict, run, i, runs_limit, epoch, 'train')

        # for name, param in my_bdq.policy_network.named_parameters():
        #     TB_1.add_histogram(name, param, epoch)
        #     TB.add_histogram(f'{name}.grad', param.grad, epoch)

        if (epoch) % 5 == 0 and epoch != 0:
            torch.save(my_bdq.policy_network.state_dict(), os.path.join(f'MEAN_reward_epoch_{epoch+14}_lr_{lr}'))  # save model
            print('********** saved model ************')

        # Only need 1 baseline epoch
        # if not act:
        #     continue

    # -- Save Model (don't save benchmark model if only 1 epoch)
    if experiment_params_dict['save_model'] and not (epoch == 0 and experiment_params_dict['run_benchmark']):
        torch.save(my_bdq.policy_network.state_dict(), os.path.join('lstm_50_epoch_model'))  # save model

    if i >= runs_limit - 1:
        "Breaking from loop"
        break
