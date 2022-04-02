import os
import gc
from typing import Union

from emspy import EmsPy
from bca import Agent_TB, ModelManager, RunManager, mdp_manager, paths_config
from bca import BranchingDQN, BranchingDQN_RNN


# ------------------------------------------------ Run Study ------------------------------------------------
my_bdq = run_manager.create_bdq(run, rnn=run.rnn)


    def run_experiment(run: RunManager.Run,
                       run_manager: RunManager,
                       bdq: Union[BranchingDQN, BranchingDQN_RNN],
                       idf_file_base: str,
                       idf_file_final: str,
                       epw_file: str,
                       year: int,
                       sim_type: str = 'train'):

        if sim_type == 'benchmark':
            learn = False
            act = False
            exploit = False
        elif sim_type == 'exploit':
            learn = False
            act = True
            exploit = True
        else:
            learn = True
            act = True
            exploit = False

        # - Clean Memory -
        if 'my_mdp' in locals():
            del my_mdp, my_sim, my_policy, my_memory, my_agent
            gc.collect()  # release memory

        # -- Simulation Params --
        cp = EmsPy.available_calling_points[9]  # 6-16 valid for timestep loop (9*)

        # -- INSTANTIATE MDP / CUSTOM IDF / CREATE SIM --
        my_model = ModelManager(
            mdp_manager_file=mdp_manager,
            idf_file_input=idf_file_base,
            idf_file_output=idf_file_final,
            year=year
        )
        my_model.create_custom_idf()

        # -- Create MDP & Building Sim Instance --
        my_mdp = my_model.create_mdp()
        my_sim = my_model.create_sim(my_mdp)

        # -- Instantiate RL Agent --
        my_policy = run_manager.create_policy(run)
        my_memory = run_manager.create_exp_replay(run, rnn=run.rnn)
        my_agent = run_manager.create_agent(run, my_mdp, my_sim, tb)

        # -- Set Sim Calling Point(s) & Callback Function(s) --
        my_sim.set_calling_point_and_callback_function(
            calling_point=cp,
            observation_function=my_agent.observe,
            actuation_function=my_agent.act_step_fixed_setpoints,  # Try different actuation functions
            update_state=True,
            update_observation_frequency=run.interaction_ts_frequency,
            update_actuation_frequency=run.interaction_ts_frequency,
            observation_function_kwargs={'learn': learn},
            actuation_function_kwargs={'actuate': act, 'exploit': exploit}
        )

        # --**-- Run Sim --**--
        my_sim.run_env(epw_file)
        my_sim.reset_state()