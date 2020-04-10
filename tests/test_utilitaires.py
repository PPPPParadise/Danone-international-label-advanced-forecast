# coding: utf-8
import logging
import os
import sys
import unittest
import datetime
import dateutil
from cfg.paths import DIR_TEST_DATA
from functools import reduce

import pandas as pd

# from cfg.paths import LOG_FILENAME, LOG_FORMAT
import src.forecaster.utilitaires as utils

# logging.basicConfig(filename=LOG_FILENAME % 'preprocessing', format=LOG_FORMAT, level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TestUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # Generating data / params
        cls.list_period = utils.create_list_period(201601, 202012)
        cls.horizon = 6
        cls.years_to_add = 3
        cls.combination_date = utils.get_all_combination_date(cls.list_period, cls.horizon)

        # Reading test data for il
        cls.raw_master_il = pd.read_csv(os.path.join(DIR_TEST_DATA, 'raw_master_il.csv'), parse_dates=['date'])
        cls.all_sales_il = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_all_sales_il.pkl'))
        cls.forecast_il = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_extend_forecast_il.pkl'))
        cls.pre_forecast_correction_il = pd.read_pickle(
            os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_il.pkl'))
        cls.long_il = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_reformat_il.pkl'))

        # Reading test data for dc
        cls.raw_master_dc = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'raw_master_dc.pkl'))
        cls.all_sales_dc = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_all_sales_dc.pkl'))
        cls.forecast_dc = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_extend_forecast_dc.pkl'))
        cls.pre_forecast_correction_dc = pd.read_pickle(
            os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_dc.pkl'))
        cls.long_dc = pd.read_pickle(os.path.join(DIR_TEST_DATA, 'test_reformat_dc.pkl'))

    @staticmethod
    def convert_date_format(date):
        new_date = datetime.datetime.strptime(str(date), '%Y%m')
        return new_date

    @staticmethod
    def compute_month_delta(new_date, old_date):
        return 12 * (new_date // 100 - old_date // 100) + new_date % 100 - old_date % 100

    def test_create_list_period(self):

        # 1. Checking that we have the correct number of months
        nb_dates = len(self.list_period)
        real_nb_dates = self.compute_month_delta(self.list_period[-1], self.list_period[0]) + 1
        self.assertEqual(nb_dates, real_nb_dates, msg='Wrong number of dates')

        # 2. Checking that each month is correctly spaced in time
        for old_date, new_date in zip(self.list_period[:-1], self.list_period[1:]):
            old = self.convert_date_format(old_date)
            new = self.convert_date_format(new_date)
            self.assertEqual(old + dateutil.relativedelta.relativedelta(months=1), new, msg='Not all months included')

    def test_add_period(self):

        # Checking that each month is correctly spaced in time
        for old_date, new_date in zip(self.list_period[:-1], self.list_period[1:]):
            self.assertEqual(utils.add_period(date=old_date, add=1), new_date, msg='Not spaced by a month')

    def test_get_all_combination_date(self):

        # 1. Checking that list have correct length
        dwp_list, dtp_list = self.combination_date
        self.assertEqual(len(dwp_list), len(dtp_list), msg='Not the same number of dates')

        # 2. Verifying that dates are matched with the correct horizon
        month_diff = 0
        for dwp, dtp in zip(dwp_list, dtp_list):
            month_diff = month_diff % self.horizon
            month_diff += 1
            self.assertEqual(self.compute_month_delta(dtp, dwp), month_diff, msg='Wrong horizon')

    def test_get_observed_sales(self):

        toy_df_1 = pd.DataFrame({
            'label': ['di', 'di', 'di'],
            'date_to_predict': [201801, 201802, 201803],
            'offtake': [5, 100, 5]
        })

        # Checking for proper computation
        to_be_tested = utils.get_observed_sales(toy_df_1, 'di', 2018, 2, 'date_to_predict', 'offtake')
        self.assertEqual(to_be_tested, 10, msg='Wrong computation')

    def test_extend_forecast(self):

        extended_forecast_il = utils.extend_forecast(
            self.forecast_il, self.raw_master_il, n_year_extended_horizon=self.years_to_add, di_eib_il_format=True)
        extended_forecast_dc = utils.extend_forecast(
            self.forecast_dc, self.raw_master_dc, di_eib_il_format=False, n_year_extended_horizon=self.years_to_add)

        self.assertEqual(
            (extended_forecast_il.shape[0] - self.forecast_il.shape[0]) / extended_forecast_il.sku_wo_pkg.nunique()
            , 12 * self.years_to_add, msg='Wrong number of months added for il')

        self.assertEqual(
            (extended_forecast_dc.shape[0] - self.forecast_dc.shape[0]) / extended_forecast_dc.sku.nunique()
            , 12 * self.years_to_add, msg='Wrong number of months added for dc')

        last_forecast_il = self.forecast_il.loc[
            (self.forecast_il.prediction_horizon >= 1) & (
                    self.forecast_il.prediction_horizon <= 12)].copy().reset_index(drop=True).drop(
            ['date', 'sku_wo_pkg', 'prediction_horizon'], axis=1)
        extension_il = pd.concat([extended_forecast_il, self.forecast_il]).drop_duplicates(keep=False)

        last_forecast_dc = self.forecast_dc.loc[
            (self.forecast_dc.prediction_horizon >= 1) & (
                    self.forecast_dc.prediction_horizon <= 12)].copy().reset_index(drop=True).drop(
            ['date', 'sku', 'prediction_horizon'], axis=1)
        extension_dc = pd.concat([extended_forecast_dc, self.forecast_dc]).drop_duplicates(keep=False)

        # Checking if ratios are the same for all labels for each month, on a rolling year basis
        for y in range(1, self.years_to_add + 1):
            temp_il = extension_il.loc[(extension_il.prediction_horizon > 12 * y) & (
                    extension_il.prediction_horizon <= 12 * (y + 1))].copy().reset_index(drop=True).drop(
                ['date', 'sku_wo_pkg', 'prediction_horizon'], axis=1)
            temp_dc = extension_dc.loc[(extension_dc.prediction_horizon > 12 * y) & (
                    extension_dc.prediction_horizon <= 12 * (y + 1))].copy().reset_index(drop=True).drop(
                ['date', 'sku', 'prediction_horizon'], axis=1)

            ratio_il = (temp_il / last_forecast_il).round(decimals=3).fillna(method='bfill')
            ratio_dc = (temp_dc / last_forecast_dc).round(decimals=3).fillna(method='bfill')

            self.assertEqual(len(ratio_il.drop_duplicates()), 1, msg='Different ratios for IL')
            self.assertEqual(len(ratio_dc.drop_duplicates()), 1, msg='Different ratios for DC')

            # Updating values to be tested against
            last_forecast_il = extension_il.loc[(extension_il.prediction_horizon > 12 * y) & (
                    extension_il.prediction_horizon <= 12 * (y + 1))].copy().reset_index(drop=True).drop(
                ['date', 'sku_wo_pkg', 'prediction_horizon'], axis=1)

            last_forecast_dc = extension_dc.loc[(extension_dc.prediction_horizon > 12 * y) & (
                    extension_dc.prediction_horizon <= 12 * (y + 1))].copy().reset_index(drop=True).drop(
                ['date', 'sku', 'prediction_horizon'], axis=1)

    def test_apply_forecast_correction(self):

        corrected_forecast = utils.apply_forecast_correction(self.all_sales_il,
                                                             self.pre_forecast_correction_il,
                                                             self.pre_forecast_correction_il[
                                                                 self.pre_forecast_correction_il.label == 'eib'], 'eib',
                                                             2020, 2, thrsh=0).reset_index(drop=True)

        # Testing EIB
        eib_pre_correction = self.pre_forecast_correction_il.loc[self.pre_forecast_correction_il.label == 'eib', :].copy().reset_index(drop=True)

        # Only February should be corrected
        self.assertGreater((corrected_forecast.loc[corrected_forecast.date_to_predict == 202002, 'prediction'] - eib_pre_correction.loc[eib_pre_correction.date_to_predict == 202002, 'prediction']).sum(), 0, msg='Was not corrected and should be')

        # And March shouldn't (e.g.)
        self.assertEqual((corrected_forecast.loc[corrected_forecast.date_to_predict == 202003, 'prediction'] - eib_pre_correction.loc[eib_pre_correction.date_to_predict == 202003, 'prediction']).sum(), 0, msg='Was corrected and shouldn\'t')

    def test_convert_long_to_wide(self):
        def multiply(to_be_reduced):
            return reduce(lambda x, y: x * y, to_be_reduced)

        self.assertEqual(multiply(self.long_il.shape), 2 * multiply(utils.convert_long_to_wide(self.long_il, self.raw_master_il, True).shape), msg='Lost data for il')
        self.assertEqual(multiply(self.long_dc.shape), utils.convert_long_to_wide(self.long_dc, self.raw_master_dc, False).shape[0] * (utils.convert_long_to_wide(self.long_dc, self.raw_master_dc, False).shape[1] - 1), msg='Lost data for dc')


if __name__ == '__main__':
    unittest.main()
