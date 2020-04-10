# coding: utf-8
import logging
import os
import unittest

import numpy as np
import pandas as pd
import yaml

from cfg.paths import DIR_CFG, DIR_CACHE
from cfg.paths import LOG_FILENAME
from src.data_wrangling.RawMasterDC import SELECTED_SKUS as SELECTED_SKU_DC
from src.data_wrangling.RawMasterIL import EXCLUDED_SKUS as EXCLUDED_SKUS_IL
from src.scenario import *
from src.scenario import F_DN_OFT_VAL, F_DN_MEA_DAT, F_CONNECT, F_RESULTS, F_DATABASE, F_TABLE, \
    F_DN_FRC_USR_NAM_DSC, F_DN_FRC_CRE_DAT, F_DN_FRC_MDF_DAT, F_DN_DATE_FMT
from src.scenario.Scenario import Scenario
from src.utils.impala_utils import ImpalaUtils
from src.utils.logging import setup_loggers

logger = logging.getLogger(__name__)
setup_loggers('test_data_pipeline', log_filepath=LOG_FILENAME, level=logging.INFO)

COMP_SELLOUT = 'comp_sellout'
COMP_SELLIN = 'comp_sellin'
COMP_OFFTAKE = 'comp_offtake'


class TestDataPipeline(unittest.TestCase):

    # @unittest.skip('Skipped')
    def test_check_il_raw_to_postproc(self):
        # load data obtained from run.demand_forecast_il
        data_il_raw_master: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'raw_master_il.pkl'))
        data_il_output_postproc: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'il_actual_split.pkl'))

        first_forecast_month = data_il_output_postproc[F_AF_DATE].max()

        # complement data_il_raw_master to enable merging
        data_il_raw_master[F_AF_SUPERLABEL] = 'IL'
        id_vars = [F_AF_DATE] + [col for col in V_AF_LOWEST_GRANULARITY
                                 if col not in {F_AF_CHANNEL, F_AF_LABEL, F_AF_SKU_WITH_PKG}]
        data_il_raw_master_melted_offtake = pd.melt(
            data_il_raw_master,
            var_name=F_AF_LABEL,
            value_name=F_AF_OFFTAKE,
            value_vars=[F_AF_OFFTAKE_IL, F_AF_OFFTAKE_EIB, F_AF_OFFTAKE_DI],
            id_vars=id_vars
        ).replace({F_AF_LABEL: {F_AF_OFFTAKE_DI: V_DI, F_AF_OFFTAKE_EIB: V_EIB, F_AF_OFFTAKE_IL: V_IL}}, inplace=False)
        data_il_raw_master_melted_sellin = pd.melt(
            data_il_raw_master,
            var_name=F_AF_LABEL,
            value_name=F_AF_SELLIN,
            value_vars=[F_AF_SELLIN_IL, F_AF_SELLIN_DI, F_AF_SELLIN_EIB],
            id_vars=id_vars
        ).replace({F_AF_LABEL: {F_AF_SELLIN_DI: V_DI, F_AF_SELLIN_EIB: V_EIB, F_AF_SELLIN_IL: V_IL}}, inplace=False)
        data_il_raw_master_melted_sellin_merged = pd.merge(
            left=data_il_raw_master_melted_offtake,
            right=data_il_raw_master_melted_sellin,
            on=id_vars + [F_AF_LABEL]
        )

        # complement data_il_output_postproc to enable merging later
        data_il_output_postproc.fillna(value=0, inplace=True)
        data_il_output_postproc.replace({'null': 0}, inplace=True)
        data_il_output_postproc[F_AF_SELLOUT] = data_il_output_postproc[F_AF_SELLOUT].apply(lambda x: float(x))
        data_il_output_postproc_aggd = data_il_output_postproc.groupby(
            by=[F_AF_DATE] + [col for col in V_AF_LOWEST_GRANULARITY if col not in {F_AF_CHANNEL, F_AF_SKU_WITH_PKG}]
        ).agg({
            F_AF_OFFTAKE: np.sum,
            F_AF_SELLOUT: np.sum,
            F_AF_SELLIN: np.sum,
        }).reset_index()

        # merge and compare columns
        check_df = pd.merge(
            left=data_il_raw_master_melted_sellin_merged,
            right=data_il_output_postproc_aggd,
            on=[F_AF_DATE] + [col for col in V_AF_LOWEST_GRANULARITY if col not in {F_AF_CHANNEL, F_AF_SKU_WITH_PKG}],
            suffixes=['_raw', '_postproc'],
            how='outer'
        )
        check_df.fillna(value=0.0, inplace=True)

        # remove entries in forecast time range (data in raw_master not coming from actual_split, but fcst)
        check_df = check_df.loc[check_df[F_AF_DATE] < first_forecast_month, :]

        check_df[COMP_OFFTAKE] = (check_df[F_AF_OFFTAKE + '_raw'] - check_df[F_AF_OFFTAKE + '_postproc']).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_AF_SELLIN + '_raw'] - check_df[F_AF_SELLIN + '_postproc']).abs() < 0.001

        diff: pd.DataFrame = check_df.loc[~np.any(check_df[[COMP_OFFTAKE, COMP_SELLIN]], axis=1), :]

        if len(diff.index) > 0:
            diff.to_csv(os.path.join(DIR_CACHE, 'diff_values_test_check_il_raw_to_postproc.csv'))
        self.assertEqual(len(diff.index), 0)

    # @unittest.skip('Skipped')
    def test_check_il_postproc_to_scenario(self):
        # load data obtained from run.demand_forecast_il
        data_il_output_postproc = pd.read_pickle(os.path.join(DIR_CACHE, 'il_actual_split.pkl'))
        data_il_output_scenario: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'scenario_il.pkl')).forecasts

        data_il_output_postproc \
            = data_il_output_postproc.loc[~data_il_output_postproc[F_AF_SKU_WO_PKG].isin(EXCLUDED_SKUS_IL), :]

        try:
            first_forecast_month = \
                data_il_output_scenario.loc[data_il_output_scenario[F_DN_FRC_MTH_NBR] == -2, F_DN_MEA_DAT].iloc[0]
        except IndexError:
            first_forecast_month = \
                data_il_output_scenario.loc[data_il_output_scenario[F_DN_FRC_MTH_NBR] == -1, F_DN_MEA_DAT].iloc[0]
        data_il_output_scenario = data_il_output_scenario.loc[~data_il_output_scenario[F_DN_FRC_FLG], :]

        # merge and compare columns
        check_df = pd.merge(
            left=data_il_output_postproc,
            right=data_il_output_scenario,
            left_on=[F_AF_DATE] + V_AF_LOWEST_GRANULARITY,
            right_on=[F_DN_MEA_DAT] + V_DN_LOWEST_GRANULARITY,
            how='outer'
        )

        # remove entries in forecast time range (data in raw_master not coming from actual_split, but fcst)
        check_df = check_df.loc[check_df[F_AF_DATE] < first_forecast_month, :]

        check_df[COMP_OFFTAKE] = (check_df[F_DN_OFT_VAL] - check_df[F_AF_OFFTAKE]).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_DN_SAL_INS_VAL] - check_df[F_AF_SELLIN]).abs() < 0.001
        check_df[COMP_SELLOUT] = (check_df[F_DN_SAL_OUT_VAL] - check_df[F_AF_SELLOUT]).abs() < 0.001

        diff = check_df.loc[~np.any(check_df[[COMP_OFFTAKE, COMP_SELLOUT, COMP_SELLIN]], axis=1), :]
        self.assertEqual(len(diff.index), 0)

    # @unittest.skip('Skipped')
    def test_check_dc_raw_to_postproc(self):
        # load data obtained from run.demand_forecast_il
        data_dc_raw_master: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'raw_master_dc.pkl'))
        data_dc_output_postproc: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'dc_actual_split.pkl'))

        # complement data_dc_raw_master to enable merging later
        data_dc_raw_master[F_AF_SUPERLABEL] = 'DC'
        data_dc_raw_master[F_AF_LABEL] = 'DC'
        data_dc_raw_master[F_AF_COUNTRY] = 'CN'
        data_dc_raw_master.rename(columns={F_AF_OFFTAKE_DC: F_AF_OFFTAKE, F_AF_SELLIN_DC: F_AF_SELLIN}, inplace=True)
        data_dc_raw_master.fillna(value=0, inplace=True)

        # complement data_dc_output_postproc to enable merging later
        data_dc_output_postproc.fillna(value=0, inplace=True)
        data_dc_output_postproc.replace({'null': 0}, inplace=True)
        data_dc_output_postproc[F_AF_SELLOUT] = data_dc_output_postproc[F_AF_SELLOUT].apply(lambda x: float(x))
        data_dc_output_postproc_aggd = data_dc_output_postproc.groupby(
            by=[F_AF_DATE, F_AF_SKU_WO_PKG]
        ).agg({
            F_AF_OFFTAKE: np.sum,
            F_AF_SELLOUT: np.sum,
            F_AF_SELLIN: np.sum,
        }).reset_index()

        # merge and compare columns
        check_df = pd.merge(
            left=data_dc_raw_master,
            right=data_dc_output_postproc_aggd,
            left_on=[F_AF_DATE, F_AF_SKU],
            right_on=[F_AF_DATE, F_AF_SKU_WO_PKG],
            suffixes=['_raw', '_postproc'],
            how='outer'
        )
        check_df.fillna(value=0.0, inplace=True)

        check_df[COMP_OFFTAKE] = (check_df[F_AF_OFFTAKE + '_raw'] - check_df[F_AF_OFFTAKE + '_postproc']).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_AF_SELLOUT + '_raw'] - check_df[F_AF_SELLOUT + '_postproc']).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_AF_SELLIN + '_raw'] - check_df[F_AF_SELLIN + '_postproc']).abs() < 0.001

        diff: pd.DataFrame = check_df.loc[~np.any(check_df[[COMP_OFFTAKE, COMP_SELLIN]], axis=1), :]
        if len(diff.index) > 0:
            diff.to_csv(os.path.join(DIR_CACHE, 'diff_values_test_check_dc_raw_to_postproc.csv'))
        self.assertEqual(len(diff.index), 0)

    # @unittest.skip('Skipped')
    def test_check_dc_postproc_to_scenario(self):
        # load data obtained from run.demand_forecast_dc
        data_dc_output_postproc = pd.read_pickle(os.path.join(DIR_CACHE, 'dc_actual_split.pkl'))
        data_dc_output_scenario: pd.DataFrame = pd.read_pickle(os.path.join(DIR_CACHE, 'scenario_dc.pkl')).forecasts

        data_dc_output_postproc \
            = data_dc_output_postproc.loc[data_dc_output_postproc[F_AF_SKU_WO_PKG].isin(SELECTED_SKU_DC), :]

        try:
            first_forecast_month = \
                data_dc_output_scenario.loc[data_dc_output_scenario[F_DN_FRC_MTH_NBR] == -2, F_DN_MEA_DAT].iloc[0]
        except IndexError:
            first_forecast_month = \
                data_dc_output_scenario.loc[data_dc_output_scenario[F_DN_FRC_MTH_NBR] == -1, F_DN_MEA_DAT].iloc[0]

        # merge and compare columns
        check_df = pd.merge(
            left=data_dc_output_postproc,
            right=data_dc_output_scenario.loc[~data_dc_output_scenario[F_DN_FRC_FLG], :],
            left_on=[F_AF_DATE] + V_AF_LOWEST_GRANULARITY,
            right_on=[F_DN_MEA_DAT] + V_DN_LOWEST_GRANULARITY,
            how='outer'
        )
        check_df = check_df.loc[check_df[F_DN_MEA_DAT] < first_forecast_month, :]
        check_df.fillna(value=0.0, inplace=True)
        check_df[COMP_OFFTAKE] = (check_df[F_DN_OFT_VAL] - check_df[F_AF_OFFTAKE]).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_DN_SAL_INS_VAL] - check_df[F_AF_SELLIN]).abs() < 0.001
        check_df[COMP_SELLOUT] = (check_df[F_DN_SAL_OUT_VAL] - check_df[F_AF_SELLOUT]).abs() < 0.001

        diff = check_df.loc[~np.any(check_df[[COMP_OFFTAKE, COMP_SELLOUT, COMP_SELLIN]], axis=1), :]
        self.assertEqual(len(diff.index), 0)

    # @unittest.skip('Skipped')
    def test_check_scenario_to_impala(self):
        scenario = Scenario.from_pickle('scenario_il.pkl')

        # upload scenario to Impala
        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            impala_cfg = yaml.load(ymlf)
        scenario.to_impala(
            impala_connect_cfg=impala_cfg[F_CONNECT],
            to_database=impala_cfg[F_RESULTS][F_DATABASE],
            to_table=impala_cfg[F_RESULTS][F_TABLE],
            comment='Base forecast'
        )

        # check content of scenario from Impala
        scenario_impala = Scenario.from_impala(
            user=scenario.user,
            creation_date=scenario.creation_date,
            modification_date=scenario.modification_date,
            impala_connect_cfg=impala_cfg[F_CONNECT],
            from_database=impala_cfg[F_RESULTS][F_DATABASE],
            from_table=impala_cfg[F_RESULTS][F_TABLE],
        )

        # delete scenario to avoid spamming impala
        impala_connect = ImpalaUtils.get_impala_connector(cfg=impala_cfg[F_CONNECT])
        cursor = impala_connect.cursor()
        del_req = f"""
                        DELETE FROM `{impala_cfg[F_RESULTS][F_DATABASE]}`.`{impala_cfg[F_RESULTS][F_TABLE]}`
                        WHERE `{F_DN_FRC_USR_NAM_DSC}`="{scenario_impala.user}" 
                        AND `{F_DN_FRC_CRE_DAT}`="{scenario_impala.creation_date.strftime(F_DN_DATE_FMT)}" 
                        AND `{F_DN_FRC_MDF_DAT}`="{scenario_impala.modification_date.strftime(F_DN_DATE_FMT)}"
        """
        cursor.execute(del_req)
        impala_connect.close()

        forecasts_from_file = scenario.forecasts
        forecasts_from_impala = scenario_impala.forecasts
        forecasts_from_impala.to_pickle(os.path.join(DIR_CACHE, 'tmp.pkl'))
        forecasts_from_impala = pd.read_pickle(os.path.join(DIR_CACHE, 'tmp.pkl'))
        self.assertEqual(len(forecasts_from_file.index), len(forecasts_from_impala.index))

        forecasts_from_impala[F_DN_MEA_DAT] = forecasts_from_impala[F_DN_MEA_DAT].apply(
            lambda x: int(x.strftime("%Y%m%d")))

        forecasts_from_impala[F_DN_FRC_CRE_DAT] = forecasts_from_impala[F_DN_FRC_CRE_DAT].apply(
            lambda x: x.strftime("%Y%m%d"))
        forecasts_from_impala[F_DN_FRC_MDF_DAT] = forecasts_from_impala[F_DN_FRC_MDF_DAT].apply(
            lambda x: x.strftime("%Y%m%d"))
        forecasts_from_impala[F_DN_CYC_DAT] = forecasts_from_impala[F_DN_CYC_DAT].apply(lambda x: x.strftime("%Y%m%d"))

        # merge and compare columns
        check_df = pd.merge(
            left=forecasts_from_file.loc[~forecasts_from_file[F_DN_FRC_FLG], :],
            right=forecasts_from_impala.loc[~forecasts_from_impala[F_DN_FRC_FLG], :],
            on=[F_DN_MEA_DAT] + V_DN_LOWEST_GRANULARITY,
            how='outer'
        )
        check_df[COMP_OFFTAKE] = (check_df[F_DN_OFT_VAL + '_x'] - check_df[F_DN_OFT_VAL + '_y']).abs() < 0.001
        check_df[COMP_SELLIN] = (check_df[F_DN_SAL_INS_VAL + '_x'] - check_df[F_DN_SAL_INS_VAL + '_y']).abs() < 0.001
        check_df[COMP_SELLOUT] = (check_df[F_DN_SAL_OUT_VAL + '_x'] - check_df[F_DN_SAL_OUT_VAL + '_y']).abs() < 0.001

        diff = check_df.loc[~np.any(check_df[[COMP_OFFTAKE, COMP_SELLOUT, COMP_SELLIN]], axis=1), :]
        self.assertEqual(len(diff.index), 0)


if __name__ == '__main__':
    unittest.main()
