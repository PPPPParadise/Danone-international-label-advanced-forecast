# coding: utf-8
"""
This module contains the FeatureMasterIL class which inherits from the FeatureMaster base class. The FeatureMasterIL class is required to
construct the features for the IL/EIB/DI demand forecast models.
"""
import logging

import pandas as pd

from src.data_wrangling.FeatureMaster import FeatureMaster

logger = logging.getLogger()

DEMAND_MODEL_GRANULARITY_COLS = ['stage', 'stage_3f', 'tier', 'brand', 'country', 'sku_wo_pkg']


class FeatureMasterIL(FeatureMaster):
    """This class contains functions to create the IL feature_master table"""

    def _add_il_offtake_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL offtake features')

        # add months since offtake IL/DI/EIB were first greater than 0
        df = self._add_months_since_first_gt0(df=df,
                                              value_cols=['offtake_il', 'offtake_eib', 'offtake_di'],
                                              groupby_cols=['sku_wo_pkg'],
                                              date_col='date')

        # add diff features
        df = self._add_diff_features(df=df,
                                     groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                     cols_to_diff=['offtake_di', 'offtake_eib', 'offtake_il'],
                                     periods=[1, 12, 11, 10, 9, 8, 7, 6, 5, 4])

        # date totals
        aggregation_cols = ['offtake_di', 'offtake_eib', 'offtake_il']
        df = self._add_date_totals(df=df,
                                   date_col='date',
                                   aggregation_cols=aggregation_cols)

        # shares of date totals
        df = self._add_shares_of_date_totals(df=df,
                                             date_col='date',
                                             aggregation_cols=['offtake_di', 'offtake_eib', 'offtake_il'])

        # shares of group totals
        df = self._add_shares_of_group_totals(df=df,
                                              date_col='date',
                                              groups=['stage', 'stage_3f', 'tier', 'brand', 'country', 'sku_wo_pkg'],
                                              aggregation_cols=['offtake_di', 'offtake_eib', 'offtake_il'])

        # add lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0, 1, 2],
                                       cols_to_lag=['offtake_di', 'offtake_eib', 'offtake_il'])

        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[3, 4, 5, 6, 7, 8, 9, 10, 11],
                                       cols_to_lag=['abs_diff1_offtake_di', 'abs_diff1_offtake_eib', 'abs_diff1_offtake_il'])

        # rolling features
        cols_to_roll = ['offtake_di',
                        'offtake_eib',
                        'offtake_il',
                        'share_of_stage_3f_offtake_di',
                        'share_of_stage_3f_offtake_eib',
                        'share_of_stage_3f_offtake_il',
                        'share_of_total_offtake_di',
                        'share_of_total_offtake_eib',
                        'share_of_total_offtake_il',
                        'abs_diff1_offtake_eib',
                        'abs_diff1_offtake_di',
                        'abs_diff1_offtake_il',
                        'rel_diff1_offtake_eib',
                        'rel_diff1_offtake_di',
                        'rel_diff1_offtake_il'
                        ]
        k_lst = [3, 6, 9, 12]
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=cols_to_roll,
                                     window_sizes=k_lst)

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                           cols_to_cumulate=['offtake_di', 'offtake_eib', 'offtake_il'])

        return df

    def _add_il_sellin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL sellin features')

        # add diff features
        df = self._add_diff_features(df=df,
                                     groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                     cols_to_diff=['sellin_di', 'sellin_eib', 'sellin_il'],
                                     periods=[1, 12, 11, 10, 9, 8, 7, 6, 5, 4])

        # date totals
        df = self._add_date_totals(df=df,
                                   date_col='date',
                                   aggregation_cols=['sellin_di', 'sellin_eib', 'sellin_il'])

        # shares of date totals
        df = self._add_shares_of_date_totals(df=df,
                                             date_col='date',
                                             aggregation_cols=['sellin_di', 'sellin_eib', 'sellin_il'])

        # shares of group totals
        df = self._add_shares_of_group_totals(df=df,
                                              date_col='date',
                                              groups=['stage', 'stage_3f', 'tier', 'brand', 'country', 'sku_wo_pkg'],
                                              aggregation_cols=['sellin_di', 'sellin_eib', 'sellin_il'])

        # add lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0, 1, 2],
                                       cols_to_lag=['sellin_eib', 'sellin_di', 'sellin_il'])

        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[4, 5, 6, 7, 8, 9, 10, 11],
                                       cols_to_lag=['abs_diff1_sellin_eib', 'abs_diff1_sellin_il'])

        # rolled features
        cols_to_roll = ['sellin_eib',
                        'sellin_di',
                        'sellin_il',
                        'share_of_stage_3f_sellin_eib',
                        'share_of_stage_3f_sellin_di',
                        'share_of_stage_3f_sellin_il',
                        'share_of_total_sellin_eib',
                        'share_of_total_sellin_di',
                        'share_of_total_sellin_il',
                        'abs_diff1_sellin_di',
                        'abs_diff1_sellin_eib',
                        'abs_diff1_sellin_il',
                        'rel_diff1_sellin_di',
                        'rel_diff1_sellin_eib',
                        'rel_diff1_sellin_il'
                        ]
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=cols_to_roll,
                                     window_sizes=[3, 6, 9, 12])

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                           cols_to_cumulate=['sellin_di', 'sellin_eib', 'sellin_il'])

        return df

    def _add_il_sellout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL sellout features')

        # diff features
        df = self._add_diff_features(df=df,
                                     groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                     cols_to_diff=['sellout'],
                                     periods=[1, 12, 11, 10, 9, 8, 7, 6, 5, 4])

        # date totals
        df = self._add_date_totals(df=df,
                                   date_col='date',
                                   aggregation_cols=['sellout'])

        # shares of date totals
        df = self._add_shares_of_date_totals(df=df,
                                             date_col='date',
                                             aggregation_cols=['sellout'])

        # shares of group totals
        df = self._add_shares_of_group_totals(df=df,
                                              date_col='date',
                                              groups=['stage', 'stage_3f', 'tier', 'brand', 'country', 'sku_wo_pkg'],
                                              aggregation_cols=['sellout'])

        # add lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0, 1, 2],
                                       cols_to_lag=['sellout'])

        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[11, 10, 9, 8, 7, 6, 5, 4],
                                       cols_to_lag=['abs_diff1_sellout'])

        # rolled features
        cols_to_roll = ['sellout', 'share_of_stage_3f_sellout', 'share_of_total_sellout', 'abs_diff1_sellout', 'rel_diff1_sellout']
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=cols_to_roll,
                                     window_sizes=[3, 6, 9, 12])

        # cumulated features
        df = self._add_cumulative_features(df=df,
                                           date_col='date',
                                           granularity_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                           cols_to_cumulate=['sellout'])

        return df

    def _add_il_inventory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL inventory features')

        # lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0],
                                       cols_to_lag=['retailer_inv', 'sp_inv'])

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['retailer_inv', 'sp_inv'],
                                     window_sizes=[3, 6, 9, 12])

        return df

    def _add_il_category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL category features')
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0],
                                       cols_to_lag=['value_krmb_il', 'volume_ton_il', 'total_vol', 'if_vol', 'fo_vol', 'gum_vol', 'cl_vol',
                                                    'il_vol'])

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['total_vol', 'if_vol', 'fo_vol', 'gum_vol', 'cl_vol', 'il_vol'],
                                     window_sizes=[3, 6, 9, 12])

        return df

    def _add_il_population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL population features')

        # lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0],
                                       cols_to_lag=['0to6_month_population', '6to12_month_population', '12to36_month_population'])

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['0to6_month_population', '6to12_month_population', '12to36_month_population'],
                                     window_sizes=[3, 6, 9, 12])

        return df

    def _add_il_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL price features')

        # lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0],
                                       cols_to_lag=['price', 'price_krmb_per_ton_il'])

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['price'],
                                     window_sizes=[3, 6, 9, 12])

        # relations to group averages
        df = self._add_relations_to_group_averages(df=df,
                                                   date_col='date',
                                                   groups=['stage', 'stage_3f', 'tier', 'brand', 'country', 'sku_wo_pkg'],
                                                   aggregation_cols=['price'])

        return df

    def _add_il_isa_osa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info('adding IL ISA and OSA features')
        # lag features
        df = self._add_lagged_features(df=df,
                                       groupby_cols=DEMAND_MODEL_GRANULARITY_COLS,
                                       lags=[0],
                                       cols_to_lag=['isa', 'osa'])

        # rolled features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['osa'],
                                     window_sizes=[3, 6, 9, 12])

        return df

    def _add_multisource_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['offtake_sellin_delta_il'] = df['sellin_il'] - df['offtake_il']
        df['offtake_sellin_delta_di'] = df['sellin_di'] - df['offtake_di']
        df['offtake_sellin_delta_eib'] = df['sellin_eib'] - df['offtake_eib']

        df['cumulated_offtake_sellin_delta_il'] = df.groupby(['sku_wo_pkg'], as_index=False)['offtake_sellin_delta_il'].transform(
            pd.Series.cumsum)
        df['cumulated_offtake_sellin_delta_di'] = df.groupby(['sku_wo_pkg'], as_index=False)['offtake_sellin_delta_di'].transform(
            pd.Series.cumsum)
        df['cumulated_offtake_sellin_delta_eib'] = df.groupby(['sku_wo_pkg'], as_index=False)['offtake_sellin_delta_eib'].transform(
            pd.Series.cumsum)

        # rolling features
        df = self._add_rolled_values(df=df,
                                     date_col='date',
                                     granularity=DEMAND_MODEL_GRANULARITY_COLS,
                                     value_cols=['offtake_sellin_delta_eib', 'offtake_sellin_delta_di', 'offtake_sellin_delta_il'],
                                     window_sizes=[3, 6, 9, 12])

        return df

    def _make_feature_master(self) -> pd.DataFrame:
        logger.info('assembling FeatureMasterIL')

        feature_master = self._raw_master.copy(deep=True)
        feature_master = self._add_il_offtake_features(feature_master)
        feature_master = self._add_il_sellin_features(feature_master)
        feature_master = self._add_il_sellout_features(feature_master)
        feature_master = self._add_il_inventory_features(feature_master)
        feature_master = self._add_il_category_features(feature_master)
        feature_master = self._add_il_population_features(feature_master)
        feature_master = self._add_il_price_features(feature_master)
        feature_master = self._add_il_isa_osa_features(feature_master)
        feature_master = self._add_date_features(feature_master)
        feature_master = self._add_multisource_features(feature_master)
        feature_master = self._encode_categorical_variables(feature_master, ['sku_wo_pkg', 'brand', 'country', 'stage', 'stage_3f'])
        feature_master = feature_master.sort_values(DEMAND_MODEL_GRANULARITY_COLS + ['date'], ascending=True).copy(deep=True)
        feature_master = self.drop_raw_cols(feature_master)
        self.check_feature_master(feature_master=feature_master)

        return feature_master

    @staticmethod
    def check_feature_master(feature_master: pd.DataFrame) -> None:
        """ Check whether there are duplicate columns in the feature master

        :param feature_master: Pandas dataframe containing feature master
        """
        assert len(feature_master.columns) == len(feature_master.columns.unique()), 'no duplicate columns'

    @staticmethod
    def drop_raw_cols(feature_master: pd.DataFrame) -> pd.DataFrame:
        """ Drops the raw data columns from the feature master.

        :param feature_master: Feature master with raw data columns
        :return: Original pandas input dataframe without raw data columns
        """
        cols_to_drop = ['offtake_di',
                        'offtake_eib',
                        # 'offtake_il_after_og',
                        # 'offtake_il_before_og',
                        'sellin_eib',
                        'sellin_di',
                        'retailer_inv',
                        # 'retailer_inv_month',
                        'sellout',
                        'sp_inv',
                        # 'sp_inv_month',
                        'isa',
                        'osa',
                        'price',
                        'price_krmb_per_ton_il',
                        'value_krmb_il',
                        'volume_ton_il',
                        'total_vol',
                        'if_vol',
                        'fo_vol',
                        'gum_vol',
                        'cl_vol',
                        'il_vol',
                        '0to6_month_population',
                        '6to12_month_population',
                        '12to36_month_population',
                        'offtake_il',
                        'sellin_il']

        return feature_master.drop(cols_to_drop, axis=1)
