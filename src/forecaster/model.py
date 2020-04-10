from abc import ABCMeta, abstractmethod

import pandas as pd


class Model(metaclass=ABCMeta):

    def __init__(self, data: pd.DataFrame):
        self.sales_danone = data
        self.raw_master = data.copy()

        # Conversion of date into int format
        self.sales_danone['calendar_yearmonth'] = pd.to_datetime(self.sales_danone['date']).dt.year.astype(
            str) + pd.to_datetime(self.sales_danone['date']).dt.month.astype(str).str.zfill(2)
        self.sales_danone['calendar_yearmonth'] = self.sales_danone['calendar_yearmonth'].astype(int)

        self.all_sales = None
        self.feature_importance = None
        self.preformat_table()

    @abstractmethod
    def preformat_table(self):
        """ Abstract method, used to ensure proper format regarding predited label
        """
        pass

    @abstractmethod
    def create_all_features(self, dwp, dtp):
        """ Abstract method, used to create features to be fed into the model
        :param dwp: List of dates when predicting
        :param dtp: List of dates to predict
        :return: DataFrame with features to be fed into the model
        """
        pass

    @abstractmethod
    def correct_fc(self, res, month_to_correct, thrsh):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return:
        """
        pass

    @abstractmethod
    def forecast_since_date_at_horizon(self, date_start, horizon, params):
        pass

    @abstractmethod
    def recreate_past_forecasts(self, list_dwps, params, horizon):
        pass
