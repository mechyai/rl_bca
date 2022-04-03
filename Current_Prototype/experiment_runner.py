import datetime
import os
import gc
import time

import torch

from emspy import EmsPy
from bca import RunManager, TensorboardManager
from bca import mdp_manager, _paths_config, experiment_manager

year = mdp_manager.year
model_span = 'Test'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 2,
    'load_model': r''
}

run_modification = [5e-3, 1e-3]  # , 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
idf_file_base = os.path.join(bem_folder, f'IdfFiles/BEM_V1_{year}_{model_span}')
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')
# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')
# Experiment Folder
exp_name = 'Test'
exp_folder = f'Experiments/{exp_name}_{datetime.datetime.now().strftime("%y%m%d-%H%M")}'

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# --- Study Parameters ---
run_manager = RunManager()
run = run_manager.selected_params
run_limit = len(run_modification)

# ----------------------------------------------------- Run Study -----------------------------------------------------

# -- Create DQN Model --
my_bdq = run_manager.create_bdq(run, rnn=run.rnn)
# Load model, if desired
if experiment_params_dict['load_model']:
    my_bdq.import_model(experiment_params_dict['load_model'])

# --- Run Baseline Once ---

run_type = 'benchmark'
my_tb = TensorboardManager(
    run_manager,
    name_path=os.path.join(exp_folder, f'{model_span}_BASELINE')
)

baseline_agent = experiment_manager.run_experiment(
    run=run,
    run_manager=run_manager,
    bdq=my_bdq,
    tensorboard_manager=my_tb,
    idf_file_base=idf_file_base + '_Baseline.idf',
    idf_file_final=idf_final_file,
    epw_file=epw_file,
    year=year,
    run_type=run_type,
)
my_tb.record_epoch_results(
    agent=baseline_agent,
    experimental_params=experiment_params_dict,
    run=run,
    run_count=0,
    run_limit=run_limit,
    epoch=0,
    run_type=run_type
)

# ---------------------------------------------------- Run Training ----------------------------------------------------

for run_num, param_value in enumerate(run_modification):

    if run_modification:
        # Change specific param for run
        run = run._replace(learning_rate=param_value)
        my_bdq.change_learning_rate_discrete(param_value)

    for epoch in range(experiment_params_dict['epochs']):
        print(f'\n\nRun {run_num} of {run_limit}, Epoch {epoch} of {experiment_params_dict["epochs"]}\n{run}\n\n')

        # ---- Tensor Board ----
        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_epoch{epoch + 1}-'
                                   f'{experiment_params_dict["epochs"]}_TRAIN')
        )

        run_type = 'train'
        baseline_agent = experiment_manager.run_experiment(
            run=run,
            run_manager=run_manager,
            bdq=my_bdq,
            tensorboard_manager=my_tb,
            idf_file_base=idf_file_base + '.idf',
            idf_file_final=idf_final_file,
            epw_file=epw_file,
            year=year,
            run_type=run_type,
        )
        my_tb.record_epoch_results(
            agent=baseline_agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=0,
            run_limit=run_limit,
            epoch=0,
            run_type=run_type
        )

    # -- Save Model --
    if experiment_params_dict['epochs'] > 0:
        torch.save(my_bdq.policy_network.state_dict(),
                   os.path.join(exp_folder, f'bdq_runs_{run_num}_epochs_{experiment_params_dict["epochs"]}_lr_{param}'))
        print('********** Saved Model ************')

    # --------------------------------------------------- Run Testing --------------------------------------------------
    param = run.learning_rate
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num + 1}-{run_limit}_lr_{param}_epoch{epoch + 1}-'
                               f'{experiment_params_dict["epochs"]}_EXPLOIT')
    )

    run_type = 'exploit'
    baseline_agent = experiment_manager.run_experiment(
        run=run,
        run_manager=run_manager,
        bdq=my_bdq,
        tensorboard_manager=my_tb,
        idf_file_base=idf_file_base + '.idf',
        idf_file_final=idf_final_file,
        epw_file=epw_file,
        year=year,
        run_type=run_type,
    )
    my_tb.record_epoch_results(
        agent=baseline_agent,
        experimental_params=experiment_params_dict,
        run=run,
        run_count=0,
        run_limit=run_limit,
        epoch=0,
        run_type=run_type
    )
