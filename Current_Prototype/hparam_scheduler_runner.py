import datetime
import os
import shutil
import time

import torch

from emspy import EmsPy
from bca import MDP
from bca_manager import RunManager, TensorboardManager
from bca_manager import _paths_config, experiment_manager

# -------------------------------------------------- INPUT --------------------------------------------------

year = MDP.year
train_month_start = 'April'
train_month_end = 'April'
train_day_start = 23
train_day_end = None

test_month_start = 'May'
test_month_end = 'May'
test_day_start = 14
test_day_end = None

model_name = 'BEM_5z_2A_Base_Testbed_no_ventilation_oa1_STRAIGHT.osm'
# model_name = 'BEM_5z_2A_Base_Test.osm'

exp_name = 'act5_duel_DQN_RNN_overfit_rtp_weight_comfort_pnorm_2nd'
# exp_name = 'test'

reporting_freq = 25

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 50,
    'run_index_start': 0,
    'run_index_limit': 100,
    'skip_baseline': False,
    'experiment_desc': '',
    'print_values': False
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
exp_root = os.path.join(_paths_config.repo_root, 'Current_Prototype/HparamTest')
exp_name = f'{datetime.datetime.now().strftime("%y%m%d-%H%M")}_{exp_name}'
exp_folder = os.path.join(exp_root, f'{exp_name}')
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)
train_period = train_month_start + '_' + train_month_end
test_period = test_month_start + '_' + test_month_end

# ----------------------------------------------------- Run Study -----------------------------------------------------

# --- Study Parameters ---
run_manager = RunManager()
run_start = experiment_params_dict['run_index_start']
run_limit = experiment_params_dict['run_index_limit']

# runs = run_manager.shuffle_runs()
runs = run_manager.runs
runs = runs[run_start: run_limit]

# -------------------------------------------------- RUN BENCHMARK --------------------------------------------------

if not experiment_params_dict['skip_baseline']:
    print('\n********** Baseline **********\n')

    run_type = 'benchmark'
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder, f'_{train_period}_BASELINE')
    )

    # -- Create DQN Model --
    # Just for benchmark, not used
    _ = runs[0]
    benchmark_bdq = run_manager.create_bdq(_)

    baseline_agent = experiment_manager.run_experiment(
        run=_,
        run_manager=run_manager,
        bdq=benchmark_bdq,
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
        run=_,
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
        run=_,
        run_manager=run_manager,
        bdq=benchmark_bdq,
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
        run=_,
        run_count=0,
        run_limit=run_limit,
        epoch=0,
        run_type=run_type
    )

# ---------------------------------------------------- Run Training ----------------------------------------------------

for run_num, run in enumerate(runs):

    run_num = run_num + experiment_params_dict['run_index_start']

    # Create new BDQ model
    my_bdq = run_manager.create_bdq(run)
    my_memory = run_manager.create_replay_memory(run)

    start_step = 0
    continued_params_dict = {'epsilon_start': run.eps_start}
    if run.PER:
        continued_params_dict = {**continued_params_dict, **{'alpha_start': run.alpha_start,
                                                             'betta_start': run.betta_start}}
    # ---- Tensor Board ----
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num}-{run_limit}_TRAIN_'
                               f'{experiment_params_dict["epochs"]}_{train_period}')
    )

    for epoch in range(experiment_params_dict['epochs']):

        print(f'\nRun {run_num} of {run_limit}, Epoch {epoch + 1} of {experiment_params_dict["epochs"]}\n{run}\n')

        # Update lr tracking
        run = run._replace(learning_rate=my_bdq.lr_scheduler.optimizer.param_groups[0]['lr'])

        print('\n********** Train **********\n')
        time_start = time.time()

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
            continued_parameters=continued_params_dict,
            print_values=experiment_params_dict['print_values']
        )
        my_tb.record_epoch_results(
            agent=my_agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=run_num,
            run_limit=run_limit,
            epoch=epoch,
            run_type=run_type
        )

        start_step = my_agent.current_step
        continued_params_dict = my_agent.save_continued_params()
        my_memory.reset_between_episode()

        # LR Scheduler
        if run.lr_scheduler:
            my_bdq.lr_scheduler.step(my_agent.loss_total)

        time_train = round(time_start - time.time(), 2) / 60

        # -- Save Model Intermediate --
        if ((epoch + 1) % reporting_freq == 0 or epoch == experiment_params_dict['epochs'] - 1) and epoch != 0:
            print('\n********** Saved Model ************\n')
            model_name = f'_bdq_model_run_{run_num}_epoch_{epoch}_of_{experiment_params_dict["epochs"]}'
            torch.save(my_bdq.policy_network.state_dict(),
                       os.path.join(exp_folder, model_name))

    # ------------------------------------------------- Run Testing ------------------------------------------------

    # -- Tensorboard --
    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num}-{run_limit}_EXPLOIT_epoch{epoch + 1}-'
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
        run_count=run_num,
        run_limit=run_limit,
        epoch=epoch,
        run_type=run_type
    )

    # Save SQL
    shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype/out/eplusout.sql'), exp_folder)
    time.sleep(1)
    os.rename(os.path.join(exp_folder, 'eplusout.sql'),
              os.path.join(exp_folder, f'_{train_period}_run_{run_num}_ep{epoch + 1}_EXPLOIT_SQL.sql'))

    print('\n********** Test **********\n')

    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder,
                               f'run_{run_num}-{run_limit}_TEST_epoch{epoch + 1}-'
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
        run_count=run_num,
        run_limit=run_limit,
        epoch=epoch,
        run_type=run_type
    )

    # Save SQL
    shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype/out/eplusout.sql'), exp_folder)
    time.sleep(1)
    os.rename(os.path.join(exp_folder, 'eplusout.sql'),
              os.path.join(exp_folder, f'_{test_period}_run_{run_num}_ep{epoch + 1}_TEST_SQL.sql'))

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
