import datetime
import os

import torch

from emspy import EmsPy

from bca import MDP

from bca_manager import RunManager, TensorboardManager
from bca_manager import _paths_config, experiment_manager


year = MDP.year
month_start = 'June'
month_end = 'June'
day_start = 1
day_end = 3

exp_name = 'testing'
test_name = f'{datetime.datetime.now().strftime("%y%m%d-%H%M")}'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 1,
    'load_model': r''
}

run_modification = [5e-3]  #, 5e-5, 1e-5, 5e-6, 1e-6]

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
osm_base = os.path.join(bem_folder, 'OpenStudioModels/BEM_5z_2A_Base_Test.osm')
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')
# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')
# Experiment Folder
exp_folder = f'Experiments/{exp_name}'

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# --- Study Parameters ---
run_manager = RunManager()
run = run_manager.selected_params
run_limit = len(run_modification)

# ----------------------------------------------------- Run Study -----------------------------------------------------

# -- Create DQN Model --
my_bdq = run_manager.create_bdq(run)
# Load model, if desired
if experiment_params_dict['load_model']:
    my_bdq.import_model(experiment_params_dict['load_model'])

# ---------------------------------------------------- Run Training ----------------------------------------------------

for run_num, param_value in enumerate(run_modification):
    for epoch in range(experiment_params_dict['epochs']):

        # ---- Tensor Board ----
        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, test_name)
        )

        run_type = 'train'
        agent = experiment_manager.run_experiment(
            run=run,
            run_manager=run_manager,
            bdq=my_bdq,
            tensorboard_manager=my_tb,
            osm_file=osm_base,
            idf_file_final=idf_final_file,
            epw_file=epw_file,
            year=year,
            start_month=month_start,
            end_day=day_end,
            run_type=run_type,
        )
        my_tb.record_epoch_results(
            agent=agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=0,
            run_limit=run_limit,
            epoch=0,
            run_type=run_type
        )
