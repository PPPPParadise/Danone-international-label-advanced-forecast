# coding: utf-8
"""
This module contains the FeatureMaster base class. The FeatureMaster base class is the parent class of all FeatureMaster subclasses such as
FeatureMasterDC, FeatureMasterIL. It contains basic functionalities to calculate features (e.g. add rolling averages, lagged values, etc.)
as well as generic functions to save and load the container objects.
"""
import abc
import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger()


class FeatureMaster:
    """ Base class for the Feature Master Data object. Contains basic functionality to save / load the feature master as well as generic
    functions to calculate features.
    """

    def __init__(self, raw_master: pd.DataFrame):
        self._raw_master = raw_master
        self._feature_master = self._make_feature_master()

    @property
    def df(self) -> pd.DataFrame:
        """ Get data frame containing the feature data

        :return: Pandas dataframe containing the feature data
        """
        return self._feature_master

    @df.setter
    def df(self, _) -> Exception:
        """ Prevents this property to be set from outside the class.

        :param _: Unused.
        :return: Raises an exception.
        """
        raise AttributeError('Attribute "df" is read-only. Please create a new object.')

    @abc.abstractmethod
    def _make_feature_master(self) -> pd.DataFrame:
        """ Abstract base method that needs to be defined for all child-classes.

        :return: Pandas dataframe containing the feature data
        """
        pass

    @staticmethod
    def _add_rolled_values(df: pd.DataFrame, date_col: str, granularity: List[str], value_cols: List[str], window_sizes: List[int]):
        """ Calculate rolled features (mean, min, max) of selected columns over past k months.

        :param df: Pandas dataframe containing
        :param date_col: Name of date column
        :param granularity: List of granularity columns
        :param value_cols: List of columns for which we want to calculate rolled features
        :param window_sizes: Size of rolling window
        :return: Pandas dataframe containing rolled features
        """
        logger.info(f'calculating rolling mean/min/max of {value_cols} with window_sizes {window_sizes}')

        df = df.sort_values(date_col)

        rolled_dfs = []
        for window_size in window_sizes:
            roll_df = df.set_index(date_col).groupby(granularity, as_index=True)[value_cols].rolling(window_size, min_periods=1)
            roll_df_mean = roll_df.mean().fillna(0).reset_index()
            roll_df_min = roll_df.min().fillna(0).reset_index()
            roll_df_max = roll_df.max().fillna(0).reset_index()

            roll_df_mean.columns = [str(col) + '_mean_%dM' % window_size if col not in [date_col] + granularity else col for col in
                                    roll_df_mean.columns]
            roll_df_min.columns = [str(col) + '_min_%dM' % window_size if col not in [date_col] + granularity else col for col in
                                   roll_df_min.columns]
            roll_df_max.columns = [str(col) + '_max_%dM' % window_size if col not in [date_col] + granularity else col for col in
                                   roll_df_max.columns]

            res_df = pd.concat([roll_df_mean.set_index([date_col] + granularity),
                                roll_df_min.set_index([date_col] + granularity),
                                roll_df_max.set_index([date_col] + granularity)], axis=1)

            rolled_dfs += [res_df]

        concatenated_rolled_dfs = pd.concat(rolled_dfs, axis=1).reset_index()
        df = pd.merge(left=df,
                      right=concatenated_rolled_dfs,
                      on=['date'] + granularity,
                      how='left',
                      validate='one_to_one',
                      suffixes=(False, False))

        return df

    @staticmethod
    def _add_months_since_first_gt0(df: pd.DataFrame, date_col: str, groupby_cols: List[str], value_cols: List) -> pd.DataFrame:
        """ Add columns with number of months since value of value_col was first greater than 0

        :param df: Pandas dataframe containing raw data
        :param date_col: Name of date column
        :param groupby_cols: Columns by which we want to group data. Usually model granularity
        :param value_cols: Names of columns for which we want to calculate number of months since this value was first greater than 0
        :return: Pandas dataframe with added columns. Naming convenion: >>months_since_first_[COLUMN_NAME]_gt0<<
        """

        logger.info(f'adding months since values of {value_cols} were first >0')

        for col in value_cols:
            first_gt0_date_col_name = f'first_gt0_date_{col}'
            months_since_first_gt0_col_name = f'months_since_first_{col}_gt0'

            cond = df[col] > 0
            first_gt0_date = df.loc[cond, :].groupby(groupby_cols, as_index=False)[date_col].min()
            first_gt0_date = first_gt0_date.rename(columns={date_col: first_gt0_date_col_name})
            df = pd.merge(left=df,
                          right=first_gt0_date,
                          on=groupby_cols,
                          how='left',
                          validate='many_to_one',
                          suffixes=(False, False)
                          )

            df[months_since_first_gt0_col_name] = (df[date_col].dt.year - df[first_gt0_date_col_name].dt.year) * 12 + (
                    df[date_col].dt.month - df[first_gt0_date_col_name].dt.month)

        return df

    @staticmethod
    def _add_lagged_features(df: pd.DataFrame, groupby_cols: List[str], lags: List[int], cols_to_lag: List[str]) -> pd.DataFrame:
        """ Add time-lagged values of selected columns

        :param df: Pandas dataframe to which we want to add lagged values.
        :param groupby_cols: Columns by which we want to group data before lagging
        :param lags: List of temporal lags that we want to create values for
        :param cols_to_lag: Columns which we want to lag
        :return: Pandas dataframe with added lagged values.
        """
        logger.info(f'adding lags {lags} for columns {cols_to_lag}')

        for col in cols_to_lag:
            for lag in lags:
                df[col + f'_lag{lag}'] = df.groupby(groupby_cols)[col].apply(pd.Series.shift, periods=lag).bfill()
        return df

    @staticmethod
    def _add_diff_features(df: pd.DataFrame, groupby_cols: List[str], cols_to_diff: List[str], periods: List[int]) -> pd.DataFrame:
        """ Add absolute and relative difference features to dataframe (i.e. absolute / relative difference of current value to value of
        10 months ago

        :param df: Pandas dataframe for which we want to calculate diff features
        :param groupby_cols: List of columns by which we want to group before calculating the diff features (usually model granularity cols)
        :param cols_to_diff: List of cols that we want to differentiate
        :param periods: List of periods that we want to shift columns by for calculation of differences
        :return: Pandas dataframe with added diff features
        """

        for col in cols_to_diff:
            for p in periods:
                df[f'abs_diff{p}_{col}'] = df.groupby(groupby_cols)[col].apply(pd.Series.diff, periods=p)
                df[f'rel_diff{p}_{col}'] = ((df[f'abs_diff{p}_{col}'] / df[col]) - 1.0).fillna(0) * 100.
        return df

    @staticmethod
    def _add_cumulative_features(df: pd.DataFrame, date_col: str, granularity_cols: List[str], cols_to_cumulate: List[str]):
        """ Add cummax and cummin columns of selected columns to dataframe

        :param df: Pandas dataframe with input data
        :param date_col: Name of date column
        :param granularity_cols: List of columns that we want to group by before calculating cummax / cummin
        :param cols_to_cumulate: List of columns for which we want to calculate the cummax / cummin
        :return: Input dataframe with added cummax and cummin columns for all selected columns
        """
        for cum_col in cols_to_cumulate:
            df_cummax = df.loc[:, [date_col] + granularity_cols + [cum_col]]
            df_cummax[f'cummax_{cum_col}'] = df_cummax.groupby(granularity_cols)[cum_col].apply(pd.Series.cummax)
            df_cummax[f'cummin_{cum_col}'] = df_cummax.groupby(granularity_cols)[cum_col].apply(pd.Series.cummin)

            df = pd.merge(left=df,
                          right=df_cummax.drop([cum_col], axis=1),
                          on=['date'] + granularity_cols,
                          how='left',
                          validate='one_to_one',
                          suffixes=(False, False)
                          )

        return df

    @staticmethod
    def _add_date_totals(df: pd.DataFrame, date_col: str, aggregation_cols: List[str]) -> pd.DataFrame:
        """ Aggregate columns to date level and merge back to input table

        :param df: Pandas dataframe containing columns we want to aggregate
        :param date_col: Date column by which we want to group the data
        :param aggregation_cols: Columns which we want to aggregate to date level
        :return: Original pandas dataframe with added date totals as new columns
        """
        totals = df.loc[:, ['date'] + aggregation_cols].groupby([date_col]).sum().add_prefix('total_').reset_index()
        df = pd.merge(
            left=df,
            right=totals,
            on=date_col,
            how='left',
            validate='many_to_one',
            suffixes=(False, False)
        )
        return df

    @staticmethod
    def _add_shares_of_date_totals(df: pd.DataFrame, date_col: str, aggregation_cols: List[str]) -> pd.DataFrame:
        """ Add columns containing the share of a value of the total aggregated sum of that column (on date level)

        :param df: Pandas dataframe with input data
        :param date_col: Name of date column
        :param aggregation_cols: List of columns for which we want to calculate their shares of the column total (on date level)
        :return: Input pandas dataframe with added "share_of_total"-columns
        """
        for agg_col in aggregation_cols:
            df[f'share_of_total_{agg_col}'] = df.groupby([date_col])[agg_col].apply(lambda x: x / np.sum(x)).fillna(0)
            df[f'share_of_total_{agg_col}'] = df[f'share_of_total_{agg_col}'].fillna(0)
        return df

    @staticmethod
    def _add_shares_of_group_totals(df: pd.DataFrame, date_col: str, groups: List[str], aggregation_cols: List[str]) -> pd.DataFrame:
        """ Add columns containing the share of a value of the total aggregated sum of that column (on date + groupby-column-level)

        :param df: Pandas dataframe holding input data
        :param date_col: Name of date column
        :param groups: List of columns we want to group by (individually, together with date column)
        :param aggregation_cols: Columns we want to aggregate and calculate the shares for
        :return: Original input dataframe with added "share_of..." columns
        """
        for agg_col in aggregation_cols:
            for grp_col in groups:
                df[f'share_of_{grp_col}_{agg_col}'] = df.groupby([date_col, grp_col])[agg_col].apply(lambda x: x / np.sum(x))
                df[f'share_of_{grp_col}_{agg_col}'] = df[f'share_of_{grp_col}_{agg_col}'].fillna(0)
        return df

    @staticmethod
    def _add_relations_to_group_averages(df: pd.DataFrame, date_col: str, groups: List[str],
                                         aggregation_cols: List[str]) -> pd.DataFrame:
        """ Add columns containing information about how big a value is compared to a group average

        :param df: Pandas dataframe containing input data
        :param date_col: Name of date column
        :param groups: List of columns we want to group by (individually, together with date column)
        :param aggregation_cols: Columns we want to aggregate and calculate the relations for
        :return: Original input dataframe with added relation columns
        """
        for agg_col in aggregation_cols:

            for grp_col in groups:
                df[f'{agg_col}_rel_to_{grp_col}_average'] = df.groupby([date_col, grp_col])[agg_col].apply(lambda x: x / np.mean(x))
                df[f'{agg_col}_rel_to_{grp_col}_average'] = df[f'{agg_col}_rel_to_{grp_col}_average'].fillna(0)

        return df

    @staticmethod
    def _encode_categorical_variables(df: pd.DataFrame, cols_to_categorize: List[str]) -> pd.DataFrame:
        """ Adds encoded categorical variables to dataframe. Naming convention of added columns is "[ORIGINAL_COLUMN_NAME]_code"

        :param df: Pandas input dataframe
        :param cols_to_categorize: Cols for which we want to add encoded versions
        :return: Original input dataframe with added encoded variables
        """
        for col in cols_to_categorize:
            df[f'{col}_code'] = df[col].astype('category').cat.codes
        return df

    @staticmethod
    def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
        """ Adds date features to dataframe. The date features are:
            1. month_cat: Month extracted from date column and converted to categoric variable
            2. month_int: Month extracted from date column and converted to integer value

        :param df: Pandas input dataframe
        :return: Original input dataframe with added date features
        """
        logger.info('adding date features')
        df['month_cat'] = df['date'].dt.month.astype('category').cat.codes
        df['month_int'] = df['date'].dt.month
        return df
