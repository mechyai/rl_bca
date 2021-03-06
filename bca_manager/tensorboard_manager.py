import numpy as np
from torch.utils.tensorboard import SummaryWriter
from bca import Agent

class TensorboardManager:
    def __init__(self,
                 run_manager,
                 comment: str = '',
                 name_path: str = ''
                 ):

        if name_path:
            self.tb = SummaryWriter(log_dir=name_path, comment=comment)
        else:
            self.tb = SummaryWriter(comment=comment)

        self.run_manager = run_manager

    def record_timestep_results(self, agent: Agent):
        """Record data for each timestep from agent state."""

        # -- Learning --
        self.tb.add_scalar('Loss', agent.loss, agent.current_step)
        self.tb.add_scalar('Reward/All', agent.reward.sum(), agent.current_step)
        self.tb.add_scalar('Reward/Cumulative', agent.reward_sum.sum(), agent.current_step)
        self.tb.add_scalar('Reward/Comfort', agent.reward_component_sum[0], agent.current_step)
        self.tb.add_scalar('Reward/RTP', agent.reward_component_sum[1], agent.current_step)
        # self.tb.add_scalar('Reward/Wind', agent.reward_component_sum[2], agent.current_step)
        # Per Zone
        for zone_i in range(agent.dqn_model.action_branches):
            self.tb.add_scalar(f'PerZone/All Reward Zn{zone_i}', agent.reward_zone_sum[zone_i], agent.current_step)

        # -- Sim Data --
        self.tb.add_scalar('_SimData/RTP', agent.mdp.get_mdp_element('rtp').value, agent.current_step)
        # Hyperparam Data
        if agent.run.PER:
            self.tb.add_scalar('_SimData/PER_Betta', agent.memory.betta, agent.current_step)
            self.tb.add_scalar('_SimData/PER_Alpha', agent.memory.alpha, agent.current_step)
        self.tb.add_scalar('_SimData/LR', agent.dqn_model.optimizer.param_groups[0]['lr'])
        self.tb.add_scalar('_SimData/Epsilon', agent.epsilon, agent.current_step)
        # State Data
        # self.tb.add_histogram('State', agent.next_state_normalized)

        # Results
        # period = agent.sim.get_ems_data(['hvac_operation_sched'])
        # if period == 1:
        #     cooling_sp = 23.89
        #     heating_sp = 21.1
        # else:
        #     cooling_sp = 29.4
        #     heating_sp = 15.6
        # self.tb.add_scalar('Thermostat/Cooling_SP', cooling_sp, agent.current_step)
        self.tb.add_scalar('Thermostat/Zone0_Temp', agent.sim.get_ems_data(['zn0_temp']), agent.current_step)
        # self.tb.add_scalar('Thermostat/Heating_SP', heating_sp, agent.current_step)

        # -- Sim Results --
        self.tb.add_scalar('_Results/Comfort Dissatisfied Total', agent.comfort_dissatisfaction_total, agent.current_step)
        self.tb.add_scalar('_Results/HVAC RTP Cost Total', agent.hvac_rtp_costs_total, agent.current_step)

    def record_epoch_results(self,
                             agent: Agent,
                             experimental_params: dict,
                             run,
                             run_count: int,
                             run_limit: int,
                             epoch: int,
                             run_type: str
                             ):
        """Record data for each epoch of training."""

        # Loss & Reward
        self.tb.add_scalar('__Epoch/Loss Total', agent.loss_total, epoch)
        self.tb.add_scalar('__Epoch/Reward - All', agent.reward_sum.sum(), epoch)
        self.tb.add_scalar('__Epoch/Reward - Comfort', agent.reward_component_sum[0], epoch)
        self.tb.add_scalar('__Epoch/Reward - RTP', agent.reward_component_sum[1], epoch)
        # self.tb.add_scalar('__Epoch/Reward - Wind', agent.reward_component_sum[2], epoch)

        # Histogram
        # discomfort_histogram = np.append(agent.cold_temps_histogram_data, agent.warm_temps_histogram_data)
        # self.tb.add_histogram('Temp Discomfort per Min', discomfort_histogram)
        # self.tb.add_histogram('Cold Discomfort per Min', agent.cold_temps_histogram_data)
        # self.tb.add_histogram('Warm Discomfort per Min', agent.warm_temps_histogram_data)
        #
        # Hyperparameter
        #Pre-process hyperparameter dict
        run_hparam = run._asdict()
        for key, value in dict(run_hparam).items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                for i, value_i in enumerate(value):
                    new_key = key + f'_{i+1}'
                    run_hparam[new_key] = value_i
                del run_hparam[key]

        self.tb.add_hparams(
            hparam_dict=
            {
                **{
                    # 'run_type': run_type,
                    'run': run_count,
                    'epoch': epoch
                },
                **run_hparam
            },
            metric_dict=
            {
                'Hparam Reward - All': agent.reward_sum.sum(),
                'Hparam Reward - Comfort': agent.reward_component_sum[0],
                'Hparam Reward - RTP': agent.reward_component_sum[1],
                # 'Hparam Reward - Wind': agent.reward_component_sum[2],
            },
            # hparam_domain_discrete=
            # {
            #     **{
            #         'run_type': ['benchmark', 'train', 'exploit', 'test'],
            #         'run': list(range(run_limit)),
            #         'epoch': list(range(experimental_params['epochs']))
            #     },
            #     **self.run_manager.selected_params._asdict()
            # },
        )
