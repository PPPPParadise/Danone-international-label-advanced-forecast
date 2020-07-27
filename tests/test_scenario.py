# coding: utf-8
import logging
import os
import unittest
from datetime import datetime

import pandas as pd
import yaml
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from impala.dbapi import connect

from cfg.paths import DIR_CFG, LOG_FILENAME, DIR_CACHE, DIR_TEST_OUTPUT
from src.data_wrangling import F_DATABASE, F_CONNECT
from src.export_results.ExportToAPO import ExportToAPO
from src.scenario import MAPPING_AF_TO_DN, F_CONNECT, F_RESULTS, F_DATABASE, F_TABLE, F_DN_DATE_FMT, F_DN_OFT_VAL, \
    F_DN_CRY_COD, F_DN_PCK_SKU_COD, WRITE_BACK_INPUT_ANP, WRITE_BACK_INPUT_CATEGORY_LABEL, \
    WRITE_BACK_INPUT_CATEGORY_STAGE, WRITE_BACK_INPUT_DISTRIBUTION
from src.scenario.Scenario import Scenario
from src.utils.logging import setup_loggers
from swagger_server.models.scenario_version import ScenarioVersion

logger = logging.getLogger(__name__)
setup_loggers('test_impala', log_filepath=LOG_FILENAME, level=logging.INFO)


class TestScenario(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load configuration
        """
        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            impala_cfg = yaml.load(ymlf)
        cls.impala_connect_cfg = impala_cfg[F_CONNECT]
        cls.impala_results_dest_cfg = impala_cfg[F_RESULTS]

        cls.mapping_af_to_impala = MAPPING_AF_TO_DN

        cls.af_forecasts_il = pd.read_csv(os.path.join(DIR_CACHE, 'forecasts_IL.csv'))
        cls.af_forecasts_dc = pd.read_csv(os.path.join(DIR_CACHE, 'forecasts_DC.csv'))
        cls.actuals_il = pd.read_csv(os.path.join(DIR_CACHE, 'actuals_IL.csv'))
        cls.actuals_dc = pd.read_csv(os.path.join(DIR_CACHE, 'actuals_DC.csv'))

        cls.af_forecasts_il['date'] = pd.to_datetime(cls.af_forecasts_il['date'], format='%Y-%m-%d')
        cls.af_forecasts_dc['date'] = pd.to_datetime(cls.af_forecasts_dc['date'], format='%Y-%m-%d')
        cls.actuals_il['date'] = pd.to_datetime(cls.actuals_il['date'], format='%Y-%m-%d')
        cls.actuals_dc['date'] = pd.to_datetime(cls.actuals_dc['date'], format='%Y-%m-%d')

    @unittest.skip('Skipped')
    def test_from_af_forecasts(self):
        scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_il, af_actuals=self.actuals_il)
        self.assertFalse(scenario.forecasts.empty)
        self.assertFalse(scenario.forecasts.isnull().values.any())

    @unittest.skip('Skipped')
    def test_to_impala(self):
        data_dict = {
            'DC': {'af_forecasts': self.af_forecasts_dc, 'af_actuals': self.actuals_dc},
            'IL': {'af_forecasts': self.af_forecasts_il, 'af_actuals': self.actuals_il},
        }
        for label in {'DC', 'IL'}:
            scenario = Scenario.from_af_forecasts(af_forecasts=data_dict[label]['af_forecasts'],
                                                  af_actuals=data_dict[label]['af_actuals'])

            scenario.user = 'AF_TEST'

            scenario.calculate_tradeflow(adjusted_variable=F_DN_OFT_VAL)

            scenario.to_impala(
                impala_connect_cfg=self.impala_connect_cfg,
                to_database=self.impala_results_dest_cfg[F_DATABASE],
                to_table=self.impala_results_dest_cfg[F_TABLE]
            )

    @unittest.skip('Skipped')
    def test_from_impala(self):
        scenario = Scenario.from_impala(
            impala_connect_cfg=self.impala_connect_cfg,
            from_database=self.impala_results_dest_cfg[F_DATABASE],
            from_table=self.impala_results_dest_cfg[F_TABLE],
            user='AF_TEST',
            creation_date=datetime.strptime('2019-13-10 12:34:45', F_DN_DATE_FMT),
            modification_date=datetime.strptime('2019-13-10 12:34:45', F_DN_DATE_FMT),
        )
        self.assertFalse(scenario.forecasts.empty)

    @unittest.skip('Skipped')
    def test_to_swagger(self):
        scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_il, af_actuals=self.actuals_il)

        adjusted_metric = F_DN_OFT_VAL
        adjusted_granularity = F_DN_PCK_SKU_COD

        forecasts_swagger = scenario.to_swagger(
            metric_dn=adjusted_metric,
            granularity_dn=adjusted_granularity,
            version=ScenarioVersion(),
        )
        self.assertTrue(len(forecasts_swagger.timeseries_set) > 0)

    @unittest.skip('Skipped')
    def test_to_swagger_input(self):

        adjusted_metric_input = WRITE_BACK_INPUT_CATEGORY_LABEL
        if adjusted_metric_input in {WRITE_BACK_INPUT_ANP, WRITE_BACK_INPUT_DISTRIBUTION}:
            scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_dc, af_actuals=self.actuals_dc)
        elif adjusted_metric_input in {WRITE_BACK_INPUT_CATEGORY_STAGE, WRITE_BACK_INPUT_CATEGORY_LABEL}:
            scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_il, af_actuals=self.actuals_il)
        else:
            raise ValueError(f'Unknown adjusted input metric "{adjusted_metric_input}".')

        forecasts_swagger = scenario.to_swagger_input(
            metric_input=adjusted_metric_input,
            version=ScenarioVersion(),
        )
        self.assertTrue(len(forecasts_swagger.timeseries_set) > 0)

    # @unittest.skip('Skipped')
    def test_to_apo(self):
        scenario = Scenario.from_pickle('scenario_dc.pkl')

        scenario.calculate_tradeflow(adjusted_variable=F_DN_OFT_VAL)

        scenario.to_apo(
            apo_exporter=ExportToAPO(),
            dest_path=DIR_TEST_OUTPUT
        )

    @unittest.skip('Skipped')
    def test_deaggregate_adjustments(self):
        scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_il, af_actuals=self.actuals_il)

        adjusted_metric = F_DN_OFT_VAL
        adjusted_granularity = F_DN_CRY_COD

        adjusted_forecasts = scenario.to_swagger(  # this is the same forecast, in a different format
            metric_dn=adjusted_metric,
            granularity_dn=adjusted_granularity,
            version=ScenarioVersion(),
        )

        scenario.deaggregate_adjustments(
            adjusted_granularity_dn=adjusted_granularity,
            target_metric_dn=adjusted_metric,
            adjusted_forecasts=adjusted_forecasts,
        )

    @unittest.skip('Skipped')
    def test_calculate_tradeflow(self):
        scenario = Scenario.from_af_forecasts(af_forecasts=self.af_forecasts_il, af_actuals=self.actuals_il)

        scenario.calculate_tradeflow(adjusted_variable=F_DN_OFT_VAL)

        self.assertFalse(scenario.forecasts.empty)
        scenario.to_pickle('scenario.pkl')


if __name__ == '__main__':
    unittest.main()
