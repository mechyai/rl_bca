import datetime
import os

import torch

from emspy import EmsPy

from bca import MDP

from bca_manager import RunManager, TensorboardManager
from bca_manager import _paths_config, experiment_manager


# -------------------------------------------------- INPUT --------------------------------------------------

year = MDP.year
train_month_start = 'April'
train_month_end = 'April'
train_day_start = 1
train_day_end = 1

exp_name = '_testing'
# model_name = 'BEM_5z_2A_Base_Testbed_no_ventilation.osm'
model_name = 'BEM_5z_2A_Base_Test.osm'

run_type = 'train'
run_modification = [5e-2]  #, 5e-5, 1e-5, 5e-6, 1e-6]

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 1,
    'load_model': r'',
    'print_values': True
}

# -------------------------------------------------- START PIPELINE --------------------------------------------------

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
osm_base = os.path.join(bem_folder, 'OpenStudioModels', model_name)
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')

# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')

# Experiment Folder
test_name = f'{exp_name}_{datetime.datetime.now().strftime("%y%m%d-%H%M")}'
test_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/Experiments/_testing', test_name)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)
train_period = train_month_start + '_' + train_month_end

# --- Study Parameters ---
run_manager = RunManager()
run = run_manager.selected_params
run_limit = len(run_modification)

# ----------------------------------------------------- Run Study -----------------------------------------------------

# -- Create DQN Model --
my_bdq = run_manager.create_bdq(run)
my_memory = run_manager.create_replay_memory(run)

# Load model, if desired
if experiment_params_dict['load_model']:
    my_bdq.import_model(experiment_params_dict['load_model'])

# ---------------------------------------------------- Run Training ----------------------------------------------------

for run_num, param_value in enumerate(run_modification):

    start_step = 0
    continued_params_dict = {'epsilon_start': run.eps_start}
    if run.PER:
        continued_params_dict = {**continued_params_dict, **{'alpha_start': run.alpha_start,
                                                             'betta_start': run.betta_start}}

    for epoch in range(experiment_params_dict['epochs']):

        # ---- Tensor Board ----
        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(test_folder, test_name)
        )

        my_agent = experiment_manager.run_experiment(
            run=run,
            run_manager=run_manager,
            bdq=my_bdq,
            tensorboard_manager=my_tb,
            osm_file=osm_base,
            idf_file_final=idf_final_file,
            epw_file=epw_file,
            year=year,
            start_month=train_month_start,
            end_month=train_month_end,
            start_day=train_day_start,
            end_day=train_day_end,
            run_type=run_type,
            current_step=start_step,
            continued_parameters=continued_params_dict,
            print_values=experiment_params_dict['print_values']
        )
        my_tb.record_epoch_results(
            agent=my_agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=0,
            run_limit=run_limit,
            epoch=0,
            run_type=run_type
        )

        start_step = my_agent.current_step
        continued_params_dict = my_agent.save_continued_params()
        my_memory.reset_between_episode()
