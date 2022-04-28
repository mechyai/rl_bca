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
train_day_start = 7
train_day_end = None

test_month_start = 'May'
test_month_end = 'May'
test_day_start = None
test_day_end = None

model_name = 'BEM_5z_2A_Base_Testbed_no_ventilation_oa1.osm'

run_modification = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 5e-5, 1e-5, 5e-6, 1e-6]

# exp_name = 'Tester'
exp_name = 'act8_bdq_soft_td_1hr_reward'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 25,
    'skip_benchmark': False,
    'exploit_only': False,
    'test': True,
    'load_model': r'',
    'print_values': False,
    'experiment_desc': 'Testing new PER RNN'
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
exp_root = os.path.join(_paths_config.repo_root, 'Current_Prototype/Experiments')
exp_name = f'{datetime.datetime.now().strftime("%y%m%d-%H%M")}_{exp_name}'
if experiment_params_dict['exploit_only']:
    exp_folder = f'{exp_name}_EXPLOIT'
else:
    exp_folder = f'{exp_name}'
exp_folder = os.path.join(exp_root, exp_folder)
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

# -- Simulation Params --
cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)
train_period = train_month_start + '_' + train_month_end
test_period = test_month_start + '_' + test_month_end

# --- Study Parameters ---
run_manager = RunManager()
run = run_manager.selected_params
run_limit = len(run_modification)

# -------------------------------------------------- RUN STUDY --------------------------------------------------

# -- Create DQN Model --
my_bdq = run_manager.create_bdq(run)
my_memory = run_manager.create_replay_memory(run)

# Load model, if desired
if experiment_params_dict['load_model']:
    my_bdq.import_model(experiment_params_dict['load_model'])

# -------------------------------------------------- RUN BENCHMARK --------------------------------------------------

if not experiment_params_dict['exploit_only']:

    if not experiment_params_dict['skip_benchmark']:
        run_type = 'benchmark'
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, f'_{train_period}_BASELINE')
        )

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
            print_values=experiment_params_dict['print_values']

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
            print_values=experiment_params_dict['print_values']

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

    # -------------------------------------------------- RUN TRAINING --------------------------------------------------

    start_step = 0
    continued_params_dict = {'epsilon_start': run.eps_start}
    if run.PER:
        continued_params_dict = {**continued_params_dict, **{'alpha_start': run.alpha_start,
                                                             'betta_start': run.betta_start}}

    for run_num, param_value in enumerate(run_modification):

        # Change specific param for run
        run = run._replace(learning_rate=param_value)
        my_bdq.change_learning_rate_discrete(param_value)

        # ---- Tensor Board ----
        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_TRAIN_'
                                   f'{experiment_params_dict["epochs"]}_{train_period}')
        )

        for epoch in range(experiment_params_dict['epochs']):
            print(
                f'\nRun {run_num + 1} of {run_limit}, Epoch {epoch + 1} of {experiment_params_dict["epochs"]}\n{run}\n')

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
                run_count=0,
                run_limit=run_limit,
                epoch=0,
                run_type=run_type
            )

            # Manage looping data
            start_step = my_agent.current_step
            continued_params_dict = my_agent.save_continued_params()
            my_memory.reset_between_episode()

            time_train = round(time_start - time.time(), 2) / 60

            # -- Save Model Intermediate --
            if (epoch % 5 == 0 or epoch == experiment_params_dict['epochs'] - 1) and epoch != 0:
                print('\n********** Saved Model ************\n')
                model_name = f'bdq_runs_{run_num + 1}_epoch_{epoch}_of_{experiment_params_dict["epochs"]}_lr_{param}'
                torch.save(my_bdq.policy_network.state_dict(),
                           os.path.join(exp_folder, model_name))


        # -- Save Model --
        if experiment_params_dict['epochs'] > 0:
            print('\n********** Saved Model ************\n')
            model_name = f'bdq_runs_{run_num + 1}_epochs_{experiment_params_dict["epochs"]}_lr_{param}'
            torch.save(my_bdq.policy_network.state_dict(),
                       os.path.join(exp_folder, model_name))

        # ------------------------------------------------- RUN TESTING ------------------------------------------------

        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_EXPLOIT_epoch{epoch + 1}-'
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

        # Save SQL
        shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype/out/eplusout.sql'), exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder,
                               f'{train_period}_run_{run_num + 1}-{run_limit}_ep{epoch + 1}_EXPLOIT_SQL.sql'))

        print('\n********** Test **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_TEST_epoch{epoch + 1}-'
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

        # Save SQL
        shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype/out/eplusout.sql'), exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder, f'{test_period}_run_{run_num + 1}-{run_limit}_ep{epoch + 1}_TEST_SQL.sql'))

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

# -------------------------- No training, just run trained agent --------------------------
else:
    # Save model used to folder, with same name
    print('\n********** Saved Model ************\n')
    torch.save(my_bdq.policy_network.state_dict(),
               os.path.join(exp_folder, experiment_params_dict['load_model'].split('/')[-1]))  # get original name

    if not experiment_params_dict['test']:
        print('\n********** Exploit **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, f'__manual_{train_period}_EXPLOIT')
        )

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

        # Save SQL
        shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype\out\eplusout.sql'), exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder, f'manual_{train_month_start}_{train_month_end}_EXPLOIT_SQL.sql'))

    else:
        print('\n********** Test **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, f'__manual_{test_period}_TEST')
        )

        run_type = 'test'
        my_agent = experiment_manager.run_experiment(
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
            agent=my_agent,
            experimental_params=experiment_params_dict,
            run=run,
            run_count=0,
            run_limit=run_limit,
            epoch=0,
            run_type=run_type
        )

        # Save SQL
        shutil.copy(os.path.join(_paths_config.repo_root, r'Current_Prototype\out\eplusout.sql'), exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder, f'manual_{test_month_start}_TEST_SQL.sql'))
