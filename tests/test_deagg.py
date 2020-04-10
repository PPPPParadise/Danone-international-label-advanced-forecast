# coding: utf-8
import logging
import os
import unittest
from datetime import date

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from cfg.paths import LOG_FILENAME, DIR_TEST_DATA
from src.deagg.Deagg import Deagg
from src.scenario import *
from src.utils.logging import setup_loggers

logger = logging.getLogger(__name__)
setup_loggers('test_deagg', log_filepath=LOG_FILENAME, level=logging.INFO)

today = date.today()


class TestDeagg(unittest.TestCase):

    def test_deagg(self):
        dummy_forecasts = pd.DataFrame({
            F_DN_MEA_DAT:
                [today] * 25
                + [today + relativedelta(months=1)] * 25
                + [today + relativedelta(months=2)] * 25
                + [today + relativedelta(months=3)] * 25,
            F_DN_LV2_UMB_BRD_COD: ['DC'] * 100,
            F_DN_LV3_PDT_BRD_COD: ['AP'] * 50 + ['NC'] * 50,
            F_DN_LV5_PDT_SFM_COD: ['1'] * 25 + ['2'] * 25 + ['1'] * 25 + ['2'] * 25,
            F_DN_OFT_TRK_VAL: np.random.randint(low=100, high=200, size=100)
        })

        deagg = Deagg(forecasts_df=dummy_forecasts)

        new_values_df: pd.DataFrame = dummy_forecasts.copy(deep=True)
        new_values_df = new_values_df.groupby(
            by=[F_DN_MEA_DAT, F_DN_LV2_UMB_BRD_COD, F_DN_LV3_PDT_BRD_COD, F_DN_LV5_PDT_SFM_COD]
        ).agg({F_DN_OFT_TRK_VAL: 'sum'}).reset_index()
        new_values_df.loc[:, F_DN_OFT_TRK_VAL] = new_values_df[F_DN_OFT_TRK_VAL] + 50

        deagg.deagg(
            aggd_forecasts_new_df=new_values_df,
            metric=F_DN_OFT_TRK_VAL,
            granularity_dn=[F_DN_LV2_UMB_BRD_COD, F_DN_LV3_PDT_BRD_COD, F_DN_LV5_PDT_SFM_COD],
            inplace=True
        )

        logger.info('[OK] deagg')

    def test_aggregate_forecasts(self):
        forecasts = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'base_forecasts_df.pkl'))
        aggregated_forecasts = Deagg.aggregate_forecasts(
            forecasts=forecasts,
            granularity_dn_list=V_DN_AVAILABLE_GRANULARITIES[F_DN_LV3_PDT_BRD_COD]
        )
        self.assertTrue(len(aggregated_forecasts.index) > 0)
        logger.info('[OK] agg')


if __name__ == '__main__':
    unittest.main()
