import gc
from typing import Union

from emspy import EmsPy
from bca import BranchingDQN, BranchingDQN_RNN, MDP
from bca_manager import RunManager, ModelManager, TensorboardManager


def run_experiment(run: RunManager.Run,
                   run_manager: RunManager,
                   bdq: Union[BranchingDQN, BranchingDQN_RNN],
                   tensorboard_manager: TensorboardManager,
                   osm_file: str,
                   idf_file_final: str,
                   epw_file: str,
                   year: int,
                   start_month: Union[str, int] = 'January',
                   end_month: Union[str, int] = None,
                   start_day: int = None,
                   end_day: int = None,
                   run_type: str = 'train',
                   current_step: int = 0
                   ):
    """This runs an entire simulation for given parameters and objects."""

    if run_type == 'benchmark':
        learn = False
        act = False
        exploit = False
    elif run_type == 'exploit' or run_type == 'test':
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
        mdp_manager_file=MDP,
        osm_file=osm_file,
        idf_file_output=idf_file_final,
        year=year
    )
    my_model.set_run_period(start_month, end_month, start_day, end_day)  # change OSM run period
    my_model.osm_to_idf()  # convert to IDF
    my_model.create_custom_idf()  # append automated customizations

    # -- Create MDP & Building Sim Instance --
    my_mdp = my_model.create_mdp()
    my_sim = my_model.create_sim(my_mdp)

    # -- Instantiate RL Agent --
    my_policy = run_manager.create_policy(run)
    my_memory = run_manager.create_replay_memory(run)
    my_agent = run_manager.create_agent(run, my_mdp, my_sim, my_model, tensorboard_manager, current_step=current_step)

    # -- Set Sim Calling Point(s) & Callback Function(s) --
    my_sim.set_calling_point_and_callback_function(
        calling_point=cp,
        observation_function=my_agent.observe,
        actuation_function=my_agent.action_directory(run.actuation_function),  # Try different actuation functions
        update_state=True,
        update_observation_frequency=run.observation_ts_frequency,
        update_actuation_frequency=run.actuation_ts_frequency,
        observation_function_kwargs={'learn': learn},
        actuation_function_kwargs={'actuate': act, 'exploit': exploit}
    )

    # --**-- Run Sim --**--
    my_sim.run_env(epw_file)
    my_sim.reset_state()

    return my_agent
