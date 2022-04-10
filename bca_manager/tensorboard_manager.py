import numpy as np
from torch.utils.tensorboard import SummaryWriter


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

    def record_timestep_results(self, agent):
        """Record data for each timestep from agent state."""

        self.tb.add_scalar('Loss', agent.loss, agent.current_step)
        self.tb.add_scalar('Reward/All', agent.reward, agent.current_step)
        self.tb.add_scalar('Reward/Cumulative', agent.reward_sum, agent.current_step)
        self.tb.add_scalar('Reward/Comfort', agent.reward_component_sum[0], agent.current_step)
        self.tb.add_scalar('Reward/RTP', agent.reward_component_sum[1], agent.current_step)
        # self.tb.add_scalar('Reward/Wind', agent.reward_component_sum[2], agent.current_step)
        # Sim Data
        self.tb.add_scalar('_SimData/RTP', agent.mdp.get_mdp_element('rtp').value, agent.current_step)
        # self.tb.add_scalar('_SimData/PER_Betta', agent.memory.betta, agent.current_step)
        # Sim Results
        # self.tb.add_scalar('_Results/Comfort Dissatisfied Total', agent.comfort_dissatisfaction_total, agent.current_step)
        # self.tb.add_scalar('_Results/HVAC RTP Cost Total', agent.hvac_rtp_costs_total, agent.current_step)

    def record_epoch_results(self,
                             agent,
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
        self.tb.add_scalar('__Epoch/Reward - All', agent.reward_sum, epoch)
        self.tb.add_scalar('__Epoch/Reward - Comfort', agent.reward_component_sum[0], epoch)
        self.tb.add_scalar('__Epoch/Reward - RTP', agent.reward_component_sum[1], epoch)
        # self.tb.add_scalar('__Epoch/Reward - Wind', agent.reward_component_sum[2], epoch)

        # Histogram
        # discomfort_histogram = np.append(agent.cold_temps_histogram_data, agent.warm_temps_histogram_data)
        # self.tb.add_histogram('Temp Discomfort per Min', discomfort_histogram)
        # self.tb.add_histogram('Cold Discomfort per Min', agent.cold_temps_histogram_data)
        # self.tb.add_histogram('Warm Discomfort per Min', agent.warm_temps_histogram_data)

        # Hyperparameter
        self.tb.add_hparams(
            hparam_dict=
            {
                **{
                    'run_type': run_type,
                    'run': run_count,
                    'epoch': epoch
                },
                **run._asdict()
            },
            metric_dict=
            {
                'Hparam Reward - All': agent.reward_sum,
                'Hparam Reward - Comfort': agent.reward_component_sum[0],
                'Hparam Reward - RTP': agent.reward_component_sum[1],
                # 'Hparam Reward - Wind': agent.reward_component_sum[2],
            },
            hparam_domain_discrete=
            {
                **{
                    'run_type': ['benchmark', 'train', 'exploit', 'test'],
                    'run': list(range(run_limit)),
                    'epoch': list(range(experimental_params['epochs']))
                },
                **self.run_manager.hyperparameter_dict
            },
        )