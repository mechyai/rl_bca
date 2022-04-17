import datetime
import os
import shutil
import time

import torch

from emspy import EmsPy
from bca import MDP
from bca_manager import RunManager, TensorboardManager, _paths_config, experiment_manager

year = MDP.year
train_month_start = 'April'
train_month_end = 'April'
train_day_start = 0
train_day_end = 2

test_month_start = 'July'
test_month_end = 'July'
test_day_start = 0
test_day_end = 2

exp_name = 'new_RNN_PER_hparam'
# exp_name = 'Tester'
exp_name = f'{datetime.datetime.now().strftime("%y%m%d-%H%M")}_{exp_name}'
prepend_tb = 'PER'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 2,
    'run_index_start': 0,
    'run_index_limit': 2,
    'load_model': r'',
    'skip_baseline': True,
    'experiment_desc': 'testing PER'
}

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
osm_base = os.path.join(bem_folder, 'OpenStudioModels/BEM_5z_2A_Base_Test.osm')
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')
# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')
# Experiment Folder
exp_folder = f'HparamTest/{exp_name}'

if not os.path.exists(os.path.join(exp_folder)):
    os.makedirs(exp_folder)

train_period = train_month_start + '_' + train_month_end
test_period = test_month_start + '_' + test_month_end

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

# ----------------------------------------------------- Run Study -----------------------------------------------------

# --- Study Parameters ---
run_manager = RunManager()
runs = run_manager.runs
run_limit = experiment_params_dict['run_index_limit']
run_start = experiment_params_dict['run_index_start']

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
    name_path=os.path.join(exp_folder, f'_{train_period}_BASELINE')
)

if not experiment_params_dict['skip_baseline']:
    print('\n********** Baseline **********\n')

    baseline_agent = experiment_manager.run_experiment(
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

    # Testing Month
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder, f'_{test_period}_TEST_BASELINE')
    )

    print('\n********** Testing Baseline **********\n')

    baseline_agent = experiment_manager.run_experiment(
        run=run,
        run_manager=run_manager,
        bdq=my_bdq,
        tensorboard_manager=my_tb,
        osm_file=osm_base,
        idf_file_final=idf_final_file,
        epw_file=epw_file,
        year=year,
        start_month=test_month_start,
        end_month=test_month_end,
        start_day=test_day_start,
        end_day=test_day_end,
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

    # Create new BDQ model
    my_bdq = run_manager.create_bdq(run)
    my_memory = run_manager.create_replay_memory(run)

    start_step = 0
    for epoch in range(experiment_params_dict['epochs']):

        print(f'\nRun {run_num + 1} of {run_limit}, Epoch {epoch + 1} of {experiment_params_dict["epochs"]}\n{run}\n')

        # ---- Tensor Board ----
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_TRAIN_'
                                   f'{experiment_params_dict["epochs"]}_{train_period}')
        )

        print('\n********** Train **********\n')
        time_start = time.time()

        continued_params_dict = {
            'alpha_start': run.alpha_start,
            'betta_start': run.betta_start,
            'epsilon_start': run.eps_start
        }
        run_type = 'train'
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
            continued_parameters=continued_params_dict
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

        time_train = round(time_start - time.time(), 2) / 60

        # -- Save Model --
        if experiment_params_dict['epochs'] > 0:
            print('\n********** Saved Model ************\n')
            model_name = f'bdq_runs_{run_num + 1}_epochs_{experiment_params_dict["epochs"]}'
            torch.save(my_bdq.policy_network.state_dict(),
                       os.path.join(exp_folder, model_name))

    # ------------------------------------------------- Run Testing ------------------------------------------------

    # -- Tensorboard --
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num + 1}-{run_limit}_EXPLOIT_epoch{epoch + 1}-'
                               f'{experiment_params_dict["epochs"]}_{train_period}')
    )

    print('\n********** Exploit **********\n')

    run_type = 'exploit'
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


    print('\n********** Test **********\n')

    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num + 1}-{run_limit}_TEST_epoch{epoch + 1}-'
                               f'{experiment_params_dict["epochs"]}_{test_period}')
    )

    run_type = 'test'
    agent = experiment_manager.run_experiment(
        run=run,
        run_manager=run_manager,
        bdq=my_bdq,
        tensorboard_manager=my_tb,
        osm_file=osm_base,
        idf_file_final=idf_final_file,
        epw_file=epw_file,
        year=year,
        start_month=test_month_start,
        end_month=test_month_end,
        start_day=test_day_start,
        end_day=test_day_end,
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

    exp_file = os.path.join(exp_folder, '_exp_results_log.txt')
    # -- Save / Write Data --
    with open(exp_file, 'a+') as file:
        file.write(f'\n -----------------------------------------------------------------')
        file.write(f'\n\n Experiment Descp: {experiment_params_dict["experiment_desc"]}')
        file.write(f'\n\n Model Name: {model_name}')
        file.write(f'\n\tTime Train = {time_train} mins')
        file.write(f'\n\t*Epochs trained = Run {run_num}-{run_limit}, Epoch: {epoch + 1}')
        file.write(f'\n\t******* Cumulative Reward = {my_agent.reward_sum}')
        file.write(f'\n\t*Performance Metrics:')
        file.write(f'\n\t\tDiscomfort Metric = {my_agent.comfort_dissatisfaction_total}')
        file.write(f'\n\t\tRTP HVAC Cost Metric = {my_agent.hvac_rtp_costs_total}')
        file.write('\n\n\tHyperparameters:')
        for key, val in run._asdict().items():
            file.write(f'\n\t\t{key}: {val}')
        file.write(f'\n\nModel Architecture:\n{my_bdq.policy_network}')
