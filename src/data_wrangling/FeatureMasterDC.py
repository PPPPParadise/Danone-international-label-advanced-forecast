# coding: utf-8
"""
This module contains the FeatureMasterDC class which inherits from the FeatureMaster base class. The FeatureMasterDC class is required to
construct the features for the DC demand forecast models.
"""
import logging

import pandas as pd

from src.data_wrangling.FeatureMaster import FeatureMaster

logger = logging.getLogger()


class FeatureMasterDC(FeatureMaster):
    """This class contains functions to create the DC feature_master table"""

    def add_anp_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate ANP spending features (rolled anp values).

        :param df: Pandas dataframe to which we want to add anp spending features. Must contain anp spending raw data
        :return: Pandas dataframe with added anp spending features
        """
        logger.info('adding ANP spending features')

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=['sku'],
                                     value_cols=['anp'],
                                     window_sizes=[3, 6])

        return df

    def add_offtake_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate DC offtake features (rolled values).

        :param df: Pandas dataframe to which we want to add dc offtake features. Must contain dc offtake data
        :return: Pandas dataframe with added dc offtake features
        """
        logger.info('adding DC offtake features')

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=['sku'],
                                     value_cols=['offtake_dc'],
                                     window_sizes=[3, 6, 9])

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=['sku'],
                                           cols_to_cumulate=['offtake_dc'])

        return df

    def add_sellin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate DC sellin features (rolled values).

        :param df: Pandas dataframe to which we want to add dc sellin features. Must contain dc sellin data
        :return: Pandas dataframe with added dc offtake features
        """
        logger.info('adding DC sellin features')

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=['sku'],
                                     value_cols=['sellin_dc'],
                                     window_sizes=[3, 6])

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=['sku'],
                                           cols_to_cumulate=['sellin_dc'])

        return df

    def add_sellout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Calculate DC sellout features (rolled values).

        :param df: Pandas dataframe to which we want to add dc sellout features. Must contain dc sellout data
        :return: Pandas dataframe with added dc sellout features
        """

        logger.info('adding DC sellout features')

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=['sku'],
                                     value_cols=['sellout', 'sellout_revenue', 'sellout_price'],
                                     window_sizes=[3, 6])

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=['sku'],
                                           cols_to_cumulate=['sellout'])

        return df

    def _make_feature_master(self) -> pd.DataFrame:
        """ Constructs the feature master table

        :return: Pandas dataframe containing the feature master table
        """
        feature_master = self._raw_master.copy(deep=True)
        feature_master = self.add_offtake_features(feature_master)
        feature_master = self.add_anp_spending_features(feature_master)
        feature_master = self.add_sellin_features(feature_master)
        feature_master = self.add_sellout_features(feature_master)
        feature_master = self._encode_categorical_variables(feature_master, ['sku', 'brand'])
        return feature_master
