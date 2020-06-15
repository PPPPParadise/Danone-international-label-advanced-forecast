# coding: utf-8
import os

import numpy as np
import pandas as pd

import src.forecaster.modelil as mod
from cfg.paths import DIR_TEST_DATA
from src.forecaster.Forecaster import XGBForecaster
from src.forecaster.utilitaires import extend_forecast, convert_long_to_wide

DEFAULT_DATA_LAG_IL = 2


class ForecasterIL(XGBForecaster):

    @staticmethod
    def from_xgbparams(filename: str) -> 'ForecasterIL':
        xgb_params = XGBForecaster.load_xgb_param(filename)
        return ForecasterIL(xgb_params=xgb_params)

    def calculate_forecasts(
            self, date_start: int, horizon: int, raw_master: pd.DataFrame, data_lag: int = DEFAULT_DATA_LAG_IL
    ) -> pd.DataFrame:
        """ Defines the high level flow to calculate IL forecasts

        :param date_start: Date of 1st prediction
        :param horizon: Prediction horizon in months
        :param raw_master: Raw master data
        :param data_lag: Data lag for IL
        :return: Forecasts
        """

        self.data_lag = DEFAULT_DATA_LAG_IL
        model = mod.Modelil(raw_master)

        is_extended_forecast = False
        added_month = 0

        horizon += data_lag + 1
        # We use machine learning over the first year of forecast only, otherwise we extrapolate with a computed trend
        if horizon > 12 + data_lag + 1:
            added_month = horizon - (12 + data_lag + 1)
            is_extended_forecast = True
            horizon = 12 + data_lag + 1

        resfinal = model.forecast_since_date_at_horizon(date_start, horizon, params=self.xgb_params)
        self.feature_importance = model.feature_importance
        resfinal['horizon'] -= data_lag + 1
        resfinal.to_pickle(os.path.join(DIR_TEST_DATA, 'test_reformat_il.pkl'))
        resfinal_formatted = convert_long_to_wide(cvr=resfinal, raw_master=raw_master, di_eib_il_format=True)
        af_forecasts = resfinal_formatted[
            ['horizon', 'date', 'sku_wo_pkg', 'yhat_il_calib', 'yhat_di_calib', 'yhat_eib_calib']].rename(
            columns={'horizon': 'prediction_horizon'})

        if is_extended_forecast:
            print(f'completing forecast to {added_month + horizon}')
            af_forecasts.to_pickle(os.path.join(DIR_TEST_DATA, 'test_extend_forecast_il.pkl'))
            af_forecasts = extend_forecast(af_forecasts, raw_master, True, int(np.ceil(added_month / 12)))
            af_forecasts = af_forecasts[af_forecasts.prediction_horizon <= horizon + added_month]

        return af_forecasts
