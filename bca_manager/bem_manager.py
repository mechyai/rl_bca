import os
import datetime
from typing import Union

import openstudio

from emspy import MdpManager, BcaEnv, idf_editor

from bca import MDP

from bca_manager import _paths_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


class ModelManager:
    # -- FILE PATHS --
    # IDF File / Modification Paths
    ep_path = _paths_config.ep_path
    repo_root = _paths_config.repo_root
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

    def __init__(self, mdp_manager_file: MDP, osm_file: str, idf_file_output: str, year: int):
        """
        :param mdp_manager_file: Imported file that contains all MDP definitions
        :param osm_file: Base input OSM building model file
        :param idf_file_output: Base IDF file plus all modifications
        :param year: This is the year linked to the year of historical data
        """

        self.mdp_manager_file = mdp_manager_file
        self.idf_file_output = idf_file_output
        self.idf_file_input = None
        self.year = year

        self.osm_input = openstudio.path(osm_file)

        translator = openstudio.osversion.VersionTranslator()
        self.osm = translator.loadModel(self.osm_input).get()
        self.run_period = self.osm.getRunPeriod()

        self.start_day = None
        self.start_month = None
        self.end_day = None
        self.end_month = None

    def set_run_period(self, start_month: Union[str, int], end_month: Union[str, int] = None,
                       start_day: int = None, end_day: int = None):
        """Change OSM run period object and export to IDF."""

        if isinstance(start_month, int):
            days_of_month = {
                1: 31,
                2: 28,
                3: 31,
                4: 30,
                5: 31,
                6: 30,
                7: 31,
                8: 31,
                9: 30,
                10: 31,
                11: 30,
                12: 31
            }
        else:
            days_of_month = {
                'January': 31,
                'February': 28,
                'March': 31,
                'April': 30,
                'May': 31,
                'June': 30,
                'July': 31,
                'August': 31,
                'September': 30,
                'October': 31,
                'November': 30,
                'December': 31
            }

        # Handle variety of inputs
        if end_month is None:
            end_month = start_month
        if start_day is None:
            start_day = 1
        if end_day is None:
            end_day = days_of_month[end_month]

        if isinstance(start_month, str):
            # Convert str month back to int
            start_month = list(days_of_month.keys()).index(start_month) + 1
            end_month = list(days_of_month.keys()).index(end_month) + 1

        self.start_day = start_day
        self.start_month = start_month
        self.end_day = end_day
        self.end_month = end_month

        self.run_period.setBeginMonth(start_month)
        self.run_period.setEndMonth(end_month)
        self.run_period.setBeginDayOfMonth(start_day)
        self.run_period.setEndDayOfMonth(end_day)

    def get_sim_length(self):
        """Returns the length of days of the given configured simulation."""

        # If no custom range entered
        if self.start_month is None:
            self.start_day = self.run_period.getBeginDayOfMonth()
            self.start_month = self.run_period.getBeginMonth()
            self.end_day = self.run_period.getEndDayOfMonth()
            self.end_month = self.run_period.getEndMonth()

        start_date = datetime.datetime(self.year, self.start_month, self.start_day)
        end_date = datetime.datetime(self.year, self.end_month, self.end_day)

        return (end_date - start_date).days

    def osm_to_idf(self):
        """Convert active OSM model to IDF"""

        fwd_translator = openstudio.energyplus.ForwardTranslator()
        out_idf = fwd_translator.translateModel(self.osm)
        out_idf.save(openstudio.path(self.idf_file_output), True)

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
        # idf_file_input = self.idf_file_input
        idf_file_output = self.idf_file_output
        year = self.year

        # -- Automated IDF Modification --
        # Create final file from IDF base
        # idf_editor.append_idf(idf_file_input, os.path.join(auto_idf_folder, 'V1_IDF_modifications.idf'),
        #                       idf_file_output)
        idf_editor.append_idf(idf_file_output, os.path.join(auto_idf_folder, 'V1_IDF_modifications.idf'))

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
