import os
from abc import ABCMeta, abstractmethod

import yaml

from cfg.paths import DIR_CFG


class XGBForecaster(metaclass=ABCMeta):
    def __init__(self, xgb_params: dict):
        self.xgb_params = xgb_params
        self.data_lag = None
        self.feature_importance = None

    @staticmethod
    @abstractmethod
    def from_xgbparams(filename):
        pass

    @abstractmethod
    def calculate_forecasts(self, date_start, horizon, raw_master, data_lag):
        pass

    @staticmethod
    def load_xgb_param(filename):
        with open(os.path.join(DIR_CFG, filename), 'r') as ymlf:
            xgb_params = yaml.load(ymlf)
        return xgb_params

    def get_feature_importance(self):
        """ Returns model feature importance as a dataframe

        :return: pd.DataFrame
        """

        feature_importance = self.feature_importance
        feature_importance['horizon'] -= self.data_lag + 1
        return feature_importance
