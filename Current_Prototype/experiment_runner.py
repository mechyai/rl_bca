import datetime
import os
import shutil
import time

import torch

from emspy import EmsPy
from bca import RunManager, TensorboardManager
from bca import mdp_manager, _paths_config, experiment_manager

year = mdp_manager.year
model_span = 'Test'  # Year, May, Test
model_test = 'June'
# exp_name = 'Cool_Only_LR_Decay_Gamma_0.8_RNN_No_Shared'
exp_name = 'Tester'
exp_name = f'{datetime.datetime.now().strftime("%y%m%d-%H%M")}_{exp_name}'

# -- Experiment Params --
experiment_params_dict = {
    'epochs': 2,
    'load_model': r'',
    'exploit_only': False,
    'test': True,
    'experiment_desc': ''
}

run_modification = [5e-3, 1e-3, 5e-4] #, 5e-5, 1e-5, 5e-6]  # 1e-6]
# run_modification = [5e-3]

# -- FILE PATHS --
# IDF File / Modification Paths
bem_folder = os.path.join(_paths_config.repo_root, 'Current_Prototype/BEM')
idf_file_base = os.path.join(bem_folder, f'IdfFiles/BEM_V1_{year}_{model_span}')
idf_file_base_test = os.path.join(bem_folder, f'IdfFiles/BEM_V1_{year}_{model_test}')
idf_final_file = os.path.join(bem_folder, f'BEM_V1_{year}.idf')
# Weather Path
epw_file = os.path.join(bem_folder, f'WeatherFiles/EPW/DallasTexas_{year}CST.epw')
# Experiment Folder
if experiment_params_dict['exploit_only']:
    exp_folder = f'Experiments/{exp_name}_EXPLOIT'
else:
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

# --- Run Baseline Once ---

if not experiment_params_dict['exploit_only']:
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

    # Testing Month

    my_tb = TensorboardManager(
        run_manager,
        name_path=os.path.join(exp_folder, f'_{model_test}_TEST_BASELINE')
    )

    print('\n********** Testing Baseline **********\n')

    baseline_agent = experiment_manager.run_experiment(
        run=run,
        run_manager=run_manager,
        bdq=my_bdq,
        tensorboard_manager=my_tb,
        idf_file_base=idf_file_base_test + '.idf',
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

    # -------------------------------------------------- Run Training --------------------------------------------------

    for run_num, param_value in enumerate(run_modification):

        if run_modification:
            # Change specific param for run
            run = run._replace(learning_rate=param_value)
            my_bdq.change_learning_rate_discrete(param_value)

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
            time_start = time.time()

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

            time_train = round(time_start - time.time(), 2) / 60


        # -- Save Model --
        if experiment_params_dict['epochs'] > 0:
            print('\n********** Saved Model ************\n')
            model_name = f'bdq_runs_{run_num + 1}_epochs_{experiment_params_dict["epochs"]}_lr_{param}'
            torch.save(my_bdq.policy_network.state_dict(),
                       os.path.join(exp_folder, model_name))

        # ------------------------------------------------- Run Testing ------------------------------------------------

        param = run.learning_rate
        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_EXPLOIT_epoch{epoch + 1}-'
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

        # Save SQL
        shutil.copy(r'A:\Files\PycharmProjects\rl_bca\Current_Prototype\out\eplusout.sql', exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder, f'{model_span}_run_{run_num}-{run_limit}_ep{epoch}_EXPLOIT_SQL.sql'))

        print('\n********** Test **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder,
                                   f'run_{run_num + 1}-{run_limit}_lr_{param}_TEST_epoch{epoch + 1}-'
                                   f'{experiment_params_dict["epochs"]}_{model_test}')
        )

        run_type = 'test'
        agent = experiment_manager.run_experiment(
            run=run,
            run_manager=run_manager,
            bdq=my_bdq,
            tensorboard_manager=my_tb,
            idf_file_base=idf_file_base_test + '.idf',
            idf_file_final=idf_final_file,
            epw_file=epw_file,
            year=year,
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
        shutil.copy(r'A:\Files\PycharmProjects\rl_bca\Current_Prototype\out\eplusout.sql', exp_folder)
        time.sleep(1)
        os.rename(os.path.join(exp_folder, 'eplusout.sql'),
                  os.path.join(exp_folder, f'{model_test}_run_{run_num}-{run_limit}_ep{epoch}_TEST_SQL.sql'))

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

else:
    # Save model used to folder, with same name
    print('\n********** Saved Model ************\n')
    torch.save(my_bdq.policy_network.state_dict(),
               os.path.join(exp_folder, experiment_params_dict['load_model'].split('/')[-1]))  # get original name

    if not experiment_params_dict['test']:
        print('\n********** Exploit **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, f'__manual_{model_span}_EXPLOIT')
        )

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
    else:
        print('\n********** Test **********\n')

        my_tb = TensorboardManager(
            run_manager,
            name_path=os.path.join(exp_folder, f'__manual_{model_test}_TEST')
        )

        run_type = 'test'
        my_agent = experiment_manager.run_experiment(
            run=run,
            run_manager=run_manager,
            bdq=my_bdq,
            tensorboard_manager=my_tb,
            idf_file_base=idf_file_base_test + '.idf',
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