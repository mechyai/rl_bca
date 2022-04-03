import datetime
import os

import torch

from emspy import EmsPy
from bca import RunManager, TensorboardManager
from bca import mdp_manager, _paths_config, experiment_manager

year = mdp_manager.year
model_span = 'May'  # Year, May, Test
exp_name = 'Heat_Cool_Off'
exp_name = f'{exp_name}_{datetime.datetime.now().strftime("%y%m%d-%H%M")}'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 5,
    'run_index_start': 0,
    'run_index_limit': 50,
    'load_model': r''
}

# --- Study Parameters ---
run_manager = RunManager()
runs = run_manager.runs
run_limit = experiment_params_dict['run_index_limit']
run_start = experiment_params_dict['run_index_start']

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
idf_file_base = os.path.join(bem_folder, f'IdfFiles/BEM_V1_{year}_{model_span}')
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')
# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')
# Experiment Folder
exp_folder = f'Experiments/{exp_name}'

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# ----------------------------------------------------- Run Study -----------------------------------------------------

# -- Create DQN Model --
run = runs[0]
my_bdq = run_manager.create_bdq(run)
# Load model, if desired
if experiment_params_dict['load_model']:
    my_bdq.import_model(experiment_params_dict['load_model'])

# --- Run Baseline Once ---

run_type = 'benchmark'
my_tb = TensorboardManager(
    run_manager,
    name_path=os.path.join(exp_folder, f'_{model_span}_BASELINE')
)

print('\n********** Baseline **********\n')

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

for run_num, run in enumerate(runs):

    for epoch in range(experiment_params_dict['epochs']):
        print(f'\nRun {run_num + 1} of {run_limit}, Epoch {epoch + 1} of {experiment_params_dict["epochs"]}\n{run}\n')

        # ---- Tensor Board ----
        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_TRAIN_epoch{epoch + 1}-'
                                   f'{experiment_params_dict["epochs"]}_{model_span}')
        )

        print('\n********** Train **********\n')

        run_type = 'train'
        my_agent = experiment_manager.run_experiment(
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
            agent=my_agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=0,
            run_limit=run_limit,
            epoch=0,
            run_type=run_type
        )

    # -- Save Model --
    if experiment_params_dict['epochs'] > 0:
        print('\n********** Saved Model ************\n')
        torch.save(my_bdq.policy_network.state_dict(),
                   os.path.join(exp_folder,
                                f'bdq_runs_{run_num + 1}_epochs_{experiment_params_dict["epochs"]}'))

    # --------------------------------------------------- Run Testing --------------------------------------------------
    param = run.learning_rate
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num + 1}-{run_limit}_EXPLOIT_epoch{epoch + 1}-'
                               f'{experiment_params_dict["epochs"]}_{model_span}')
    )

    print('\n********** Exploit **********\n')

    run_type = 'exploit'
    my_agent = experiment_manager.run_experiment(
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
        agent=my_agent,
        experimental_params=experiment_params_dict,
        run=run,
        run_count=0,
        run_limit=run_limit,
        epoch=0,
        run_type=run_type
    )
