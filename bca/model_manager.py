import os

from emspy import MdpManager, BcaEnv, idf_editor
from bca import mdp_manager, paths_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


class ModelManager:
    # -- FILE PATHS --
    # IDF File / Modification Paths
    ep_path = paths_config.ep_path
    repo_root = paths_config.repo_root
    os_folder = os.path.join(repo_root, 'Current_Prototype/BEM')
    auto_idf_folder = os.path.join(os_folder, 'CustomIdfFiles/Automated')

    # -- Add CUSTOM SQL TRACKING --
    data_tracking = {  # custom tracking for actuators, (handle + unit type)
        # -- Reward --
        'reward': ('Schedule:Constant', 'Schedule Value', 'Reward Tracker', 'Dimensionless'),
        'reward_cumulative': ('Schedule:Constant', 'Schedule Value', 'Reward Cumulative', 'Dimensionless'),
        # Reward Components
        'reward_comfort': ('Schedule:Constant', 'Schedule Value', 'Reward Comfort Tracker', 'Dimensionless'),
        'reward_cumulative_comfort': ('Schedule:Constant', 'Schedule Value', 'Reward Comfort Cumulative', 'Dimensionless'),
        'reward_rtp': ('Schedule:Constant', 'Schedule Value', 'Reward RTP Tracker', 'Dimensionless'),
        'reward_cumulative_rtp': ('Schedule:Constant', 'Schedule Value', 'Reward RTP Cumulative', 'Dimensionless'),
        'reward_wind': ('Schedule:Constant', 'Schedule Value', 'Reward Wind Tracker', 'Dimensionless'),
        'reward_cumulative_wind': ('Schedule:Constant', 'Schedule Value', 'Reward Wind Cumulative', 'Dimensionless'),
        # -- Results Metrics --
        # Comfort
        'comfort': ('Schedule:Constant', 'Schedule Value', 'Comfort Tracker', 'Dimensionless'),
        'comfort_cumulative': ('Schedule:Constant', 'Schedule Value', 'Comfort Cumulative', 'Dimensionless'),
        # RTP
        'rtp_tracker': ('Schedule:Constant', 'Schedule Value', 'RTP Tracker', 'Dimensionless'),
        'rtp_cumulative': ('Schedule:Constant', 'Schedule Value', 'RTP Cumulative', 'Dimensionless'),
        # Wind
        'wind_hvac_use': ('Schedule:Constant', 'Schedule Value', 'Wind Energy HVAC Usage Tracker', 'Dimensionless'),
        'total_hvac_use': (
            'Schedule:Constant', 'Schedule Value', 'Total HVAC Energy Usage Tracker', 'Dimensionless'),
        # -- Learning --
        'loss': ('Schedule:Constant', 'Schedule Value', 'Loss Tracker', 'Dimensionless'),
    }

    def __init__(self, mdp_manager_file: mdp_manager, idf_file_input: str, idf_file_output: str, year: int):
        """
        :param mdp_manager_file: Imported file that contains all MDP definitions
        :param idf_file_input: Base input EnergyPlus IDF file
        :param idf_file_output: Base IDF file plus all modifications
        :param year: This is the year linked to the year of historical data
        """

        self.mdp_manager_file = mdp_manager_file
        self.idf_file_input = idf_file_input
        self.idf_file_output = idf_file_output
        self.year = year

    def create_sim(self, mdp: MdpManager):
        """Creates and returns BcaEnv simulation object."""
        return BcaEnv(
            ep_path=self.ep_path,
            ep_idf_to_run=self.idf_file_output,
            timesteps=60,
            tc_vars=mdp.tc_var,
            tc_intvars=mdp.tc_intvar,
            tc_meters=mdp.tc_meter,
            tc_actuator=mdp.tc_actuator,
            tc_weather=mdp.tc_weather
        )

    def create_mdp(self):
        """Creates and returns MDP object."""
        mdp = MdpManager.generate_mdp_from_tc(
            tc_intvars=self.mdp_manager_file.tc_intvars,
            tc_vars=self.mdp_manager_file.tc_vars,
            tc_meters=self.mdp_manager_file.tc_meters,
            tc_weather=self.mdp_manager_file.tc_weather,
            tc_actuators=self.mdp_manager_file.tc_actuators
        )

        # Link with ToC Actuators, remove unit types first
        for key, handle_id in self.data_tracking.items():
            mdp.add_ems_element('actuator', key, handle_id[0:3])  # exclude unit, leave handle

        return mdp

    def create_custom_idf(self):
        """
        This method take in a base input file, & adds all custom IDF components (schedule trackers, custom meters, etc.)
        to create the custom idf file
        """
        data_tracking = self.data_tracking
        auto_idf_folder = self.auto_idf_folder
        idf_file_input = self.idf_file_input
        idf_file_output = self.idf_file_output
        year = self.year

        # -- Automated IDF Modification --
        # Create final file from IDF base
        idf_editor.append_idf(idf_file_input, os.path.join(auto_idf_folder, 'V1_IDF_modifications.idf'),
                              idf_file_output)

        # Daylight savings & holidays
        # IDFeditor.append_idf(idf_final_file, f'BEM/CustomIdfFiles/Automated/TEXAS_CST_Daylight_Savings_{year}.idf')

        # Add Schedule:Files for historical data
        idf_editor.append_idf(idf_file_output,
                              os.path.join(auto_idf_folder, f'ERCOT_RTM_{year}.idf'))  # RTP
        idf_editor.append_idf(idf_file_output,
                              os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Wind.idf'))  # FMIX, wind
        idf_editor.append_idf(idf_file_output,
                              os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Solar.idf'))  # FMIX, solar
        idf_editor.append_idf(idf_file_output,
                              os.path.join(auto_idf_folder, f'ERCOT_FMIX_{year}_Total.idf'))  # FMIX, total
        # DAM 12 hr forecast
        for h in range(12):
            idf_editor.append_idf(idf_file_output,
                                  os.path.join(auto_idf_folder, f'ERCOT_DAM_12hr_forecast_{year}_{h}hr_ahead.idf'))

        # Add Custom Meters
        idf_editor.append_idf(idf_file_output, os.path.join(auto_idf_folder, 'V1_custom_meters.idf'))

        # Add Custom Data Tracking IDF Objs (reference ToC of Actuators)
        for _, handle_id in data_tracking.items():
            idf_editor.insert_custom_data_tracking(handle_id[2], idf_file_output, handle_id[3])
