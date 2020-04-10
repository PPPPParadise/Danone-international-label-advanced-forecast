# coding: utf-8
import logging
import os
import sys
import unittest
from datetime import datetime

from cfg.paths import DIR_TEST_OUTPUT
from cfg.paths import LOG_FILENAME, LOG_FORMAT
from src.export_results.ExportToAPO import ExportToAPO
from src.scenario.Scenario import Scenario

logging.basicConfig(filename=LOG_FILENAME % os.path.basename(__file__), format=LOG_FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

K_RESULTS_AF_TO_IMP = 'RESULTS_AF_TO_IMP'


class TestExportToAPO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.apo_exporter = ExportToAPO()

    # @unittest.skip
    def test_export_to_apo(self):
        scenario = Scenario.from_pickle('scenario_dc.pkl')
        ExportToAPO().export(forecasts=scenario.forecasts, dest_path=DIR_TEST_OUTPUT)

    @unittest.skip
    def test_weeks_in_month(self):

        truth_dict = {
            'January 2019': ['W 01.2019', 'W 02.2019', 'W 03.2019', 'W 04.2019', 'W 05.2019'],
            'January 2020': ['W 01.2020', 'W 02.2020', 'W 03.2020', 'W 04.2020', 'W 05.2020'],
            'January 2021': ['W 01.2021', 'W 02.2021', 'W 03.2021', 'W 04.2021'],
            'January 2022': ['W 01.2022', 'W 02.2022', 'W 03.2022', 'W 04.2022', 'W 05.2022'],
            'January 2023': ['W 01.2023', 'W 02.2023', 'W 03.2023', 'W 04.2023', 'W 05.2023'],
            'December 2018': ['W 49.2018', 'W 50.2018', 'W 51.2018', 'W 52.2018'],
            'December 2019': ['W 49.2019', 'W 50.2019', 'W 51.2019', 'W 52.2019'],
            'December 2020': ['W 50.2020', 'W 51.2020', 'W 52.2020', 'W 53.2020'],
            'December 2021': ['W 49.2021', 'W 50.2021', 'W 51.2021', 'W 52.2021'],
            'December 2022': ['W 49.2022', 'W 50.2022', 'W 51.2022', 'W 52.2022'],
            'December 2023': ['W 49.2023', 'W 50.2023', 'W 51.2023', 'W 52.2023'],
        }

        for yyyy in range(2018, 2024):
            key = f'January {yyyy}'
            if key in truth_dict:
                jan = datetime(year=yyyy, month=1, day=1)
                jan_weeks = list(ExportToAPO.weeks_in_month(date_in=jan))
                # print(f'Weeks in January {yyyy} are: {jan_weeks}')
                self.assertEqual(truth_dict[key], jan_weeks)

            key = f'December {yyyy}'
            if key in truth_dict:
                dec = datetime(year=yyyy, month=12, day=1)
                dec_weeks = list(ExportToAPO.weeks_in_month(date_in=dec))
                # print(f'Weeks in {key} are: {dec_weeks}')
                self.assertEqual(truth_dict[key], dec_weeks)

        all_weeks_2022 = []
        for m in range(1, 13):
            all_weeks_2022.extend(list(ExportToAPO.weeks_in_month(date_in=datetime(year=2022, month=m, day=1))))
        all_weeks_2022_set = set(all_weeks_2022)
        all_weeks_2022_truth = set(['W %02d.2022' % w for w in range(1, 53)])
        set_diff \
            = all_weeks_2022_truth.union(all_weeks_2022_set) - all_weeks_2022_truth.intersection(all_weeks_2022_set)
        # print(f'Weeks of 2022 are {all_weeks_2022}')
        self.assertEqual(set(), set_diff)


if __name__ == '__main__':
    unittest.main()
