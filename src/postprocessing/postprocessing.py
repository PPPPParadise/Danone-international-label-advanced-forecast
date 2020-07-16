# coding: utf-8
import datetime
import logging
import os
from typing import List

import numpy as np
import pandas as pd

from cfg.paths import DIR_CACHE
from src.data_wrangling import SELECTED_SKUS_DC
from src.scenario import *

logger = logging.getLogger(__name__)


class PostProcessing:

    @staticmethod
    def add_granularity_il(
            df_hist_di: pd.DataFrame,
            df_hist_il: pd.DataFrame,
            df_hist_il_sellin: pd.DataFrame,
            df_hist_eib: pd.DataFrame,
            df_fcst_il_all: pd.DataFrame,
            df_mapping: pd.DataFrame,
            ignore_hist_after_dt: datetime
    ):
        df_hist_di['date'] = pd.to_datetime(df_hist_di['date'])  # , format='%m/%d/%Y')
        df_hist_il['date'] = pd.to_datetime(df_hist_il['date'])  # , format='%Y-%m-%d')
        df_hist_il_sellin['date'] = pd.to_datetime(df_hist_il_sellin['date'])  # , format='%m/%d/%Y')
        df_hist_eib['date'] = pd.to_datetime(df_hist_eib['date'])  # , format='%m/%d/%Y')
        df_fcst_il_all['date'] = pd.to_datetime(df_fcst_il_all['date'])

        df_hist_di = df_hist_di.loc[df_hist_di['date'] < ignore_hist_after_dt].copy()
        df_hist_il = df_hist_il.loc[df_hist_il['date'] < ignore_hist_after_dt].copy()
        df_hist_il_sellin = df_hist_il_sellin.loc[df_hist_il_sellin['date'] < ignore_hist_after_dt].copy()
        df_hist_eib = df_hist_eib.loc[df_hist_eib['date'] < ignore_hist_after_dt].copy()

        df_hist_di.columns = df_hist_di.columns.str.lower()
        df_hist_il.columns = df_hist_il.columns.str.lower()
        df_hist_il_sellin.columns = df_hist_il_sellin.columns.str.lower()
        df_hist_eib.columns = df_hist_eib.columns.str.lower()
        df_fcst_il_all.columns = df_fcst_il_all.columns.str.lower()

        # prepare forecast format
        # df_fcst_il_all = df_fcst_il_all.loc[df_fcst_il_all[F_AF_PREDICTION_HORIZON] > 0, :]
        df_fcst_il_all.rename(columns={'sku_wo_pkg': 'sku',
                                       'yhat_il_calib': 'il',
                                       'yhat_di_calib': 'di',
                                       'yhat_eib_calib': 'eib'}, inplace=True)
        df_fcst_il_all = pd.melt(df_fcst_il_all,
                                 id_vars=['sku', F_AF_DATE, F_AF_PREDICTION_HORIZON],
                                 value_vars=['il', 'di', 'eib'])
        df_fcst_il_all.rename(columns={'variable': F_AF_LABEL,
                                       'value': F_AF_OFFTAKE}, inplace=True)

        # process DI fcst split
        def _gran_di(_df_hist_di, _df_fcst_il_all):
            # select and prepare relevant historical data
            _df_hist_di = _df_hist_di.loc[_df_hist_di['type'] == 'offtake', :]  # select offtake only
            _df_hist_di = _df_hist_di.loc[_df_hist_di['status'] == 'actual', :]  # select historical actual only
            _df_hist_di = _df_hist_di.loc[_df_hist_di['channel'] != 'Total', :]  # exclude channel totals
            _df_hist_di['quantity'] = _df_hist_di['quantity'].fillna(0)  # fill null values with 0
            # select relevant columns
            _df_hist_di = _df_hist_di.loc[:, ['sku_code', 'sku_wo_pkg', 'channel', 'quantity']]
            _df_hist_di.replace({'channel': {'ALI': 'Ali',
                                             'New channel': 'NewChannel'}}, inplace=True)
            # group by sku and channel (aggregate all dates)
            _df_hist_di = _df_hist_di.groupby(['sku_code',
                                               'sku_wo_pkg',
                                               'channel']).sum().reset_index()

            df_hist_di_sku_total = _df_hist_di.groupby(['sku_wo_pkg']).sum().reset_index()  # obtain sku-level total
            df_hist_di_sku_total.rename(columns={'quantity': 'sku_total_quantity'}, inplace=True)
            _df_hist_DI_merged = pd.merge(_df_hist_di, df_hist_di_sku_total, how='left',
                                          on='sku_wo_pkg')  # merge sku totals with granular quantity

            _df_hist_DI_merged['split_ratio'] = _df_hist_DI_merged['quantity'] / _df_hist_DI_merged[
                'sku_total_quantity']  # calculate ratio of sku total
            df_hist_DI_split = _df_hist_DI_merged.loc[:,
                               ['sku_code', 'sku_wo_pkg', 'channel', 'split_ratio']]  # select relevant columns

            # construct split table at sku_with_pkg level
            df_di_wo_pkg_total = _df_hist_di.groupby(['sku_code']).sum().reset_index()
            df_di_wo_pkg_total.rename(columns={'quantity': 'sku_code_total'}, inplace=True)
            df_di_pkg_split = pd.merge(_df_hist_di,
                                       df_di_wo_pkg_total,
                                       how='left',
                                       on='sku_code')
            df_di_pkg_split['split_ratio'] = df_di_pkg_split['quantity'] / df_di_pkg_split['sku_code_total']
            df_di_pkg_split = df_di_pkg_split.loc[:, ['sku_code', 'sku_wo_pkg', 'channel', 'split_ratio']]

            # merge with fcst data
            df_fcst_DI = df_fcst_il_all.loc[df_fcst_il_all[F_AF_LABEL] == 'di', :]
            df_fcst_DI = df_fcst_DI.loc[:, [F_AF_DATE, 'sku', F_AF_LABEL, F_AF_OFFTAKE, F_AF_PREDICTION_HORIZON]]
            df_fcst_split_DI = pd.merge(df_fcst_DI, df_hist_DI_split, how='right', left_on='sku',
                                        right_on='sku_wo_pkg')

            df_fcst_split_DI['fcst vol'] = df_fcst_split_DI[F_AF_OFFTAKE] * df_fcst_split_DI['split_ratio']
            df_fcst_split_DI = df_fcst_split_DI.loc[:,
                               [F_AF_DATE, 'sku', 'sku_code', F_AF_LABEL, F_AF_CHANNEL, 'fcst vol',
                                F_AF_PREDICTION_HORIZON]]
            df_fcst_split_DI.rename(columns={'sku': F_AF_SKU_WO_PKG,
                                             'sku_code': F_AF_SKU_WITH_PKG,
                                             'fcst vol': F_AF_OFFTAKE
                                             }, inplace=True)
            df_fcst_split_DI.replace({F_AF_LABEL: {'di': V_DI}}, inplace=True)

            return df_fcst_split_DI, df_di_pkg_split

        # process IL fcst split
        def _gran_il(_df_hist_il, _df_fcst_il_all):
            # select and prepare relevant historical data
            _df_hist_il = _df_hist_il.loc[_df_hist_il['scope'] == 'IL', :]  # select offtake only
            _df_hist_il = _df_hist_il.loc[_df_hist_il['status'] == 'actual', :]  # select historical actual only
            _df_hist_il['volume'] = _df_hist_il['volume'].fillna(0)  # fill null values with 0
            df_hist_il_channel = _df_hist_il.loc[:, ['date', 'sku_code', 'channel', 'volume']]
            df_hist_il_channel = df_hist_il_channel.groupby(['date', 'sku_code', 'channel']).sum().reset_index()
            _df_hist_il = _df_hist_il.loc[:, ['sku_code', 'channel', 'volume']]  # select relevant columns
            _df_hist_il = _df_hist_il.groupby(
                ['sku_code', 'channel']).sum().reset_index()  # group by sku and channel (aggregate all dates)
            # aggregate to SKU total level
            df_hist_il_sku_total = _df_hist_il.groupby(['sku_code']).sum().reset_index()  # obtain sku-level total
            df_hist_il_sku_total.rename(columns={'volume': 'sku_total_quantity'}, inplace=True)
            df_hist_il_merged = pd.merge(_df_hist_il, df_hist_il_sku_total, how='left',
                                         on='sku_code')  # merge sku totals with granular quantity
            # compute split ratio
            df_hist_il_merged['split_ratio'] = df_hist_il_merged['volume'] / df_hist_il_merged[
                'sku_total_quantity']  # calculate ratio of sku total
            df_hist_il_merged = df_hist_il_merged.loc[:,
                                ['sku_code', 'channel', 'split_ratio']]  # select relevant columns
            # merge with fcst data
            df_fcst_il = _df_fcst_il_all.loc[_df_fcst_il_all[F_AF_LABEL] == 'il', :]
            df_fcst_il = df_fcst_il.loc[:, [F_AF_DATE, F_AF_LABEL, 'sku', F_AF_OFFTAKE, F_AF_PREDICTION_HORIZON]]
            df_fcst_split_il = pd.merge(df_fcst_il, df_hist_il_merged, how='right', left_on='sku', right_on='sku_code')
            # apply split ratio
            df_fcst_split_il['fcst vol'] = df_fcst_split_il[F_AF_OFFTAKE] * df_fcst_split_il[
                'split_ratio']  # apply split ratio
            df_fcst_split_il = df_fcst_split_il.loc[pd.notnull(df_fcst_split_il['fcst vol']),
                               :]  # exclude SKUs that are out of scope
            df_fcst_split_il = df_fcst_split_il.loc[:, [F_AF_DATE, F_AF_LABEL, 'sku', F_AF_CHANNEL, 'fcst vol',
                                                        F_AF_PREDICTION_HORIZON]]  # select relevant columns
            df_fcst_split_il.rename(columns={'sku': F_AF_SKU_WO_PKG,
                                             'fcst vol': F_AF_OFFTAKE
                                             }, inplace=True)
            df_fcst_split_il.replace({F_AF_LABEL: {'il': V_IL}}, inplace=True)
            df_fcst_split_il[F_AF_SKU_WITH_PKG] = df_fcst_split_il[F_AF_SKU_WO_PKG]

            return df_fcst_split_il, df_hist_il_merged, df_hist_il_channel

        # extract eib fcst (no granularity split)
        def _prep_eib_fcst(_df_fcst_il_all):
            df_fcst_eib = _df_fcst_il_all.loc[df_fcst_il_all[F_AF_LABEL] == 'eib', :]
            df_fcst_eib = df_fcst_eib.loc[:, [F_AF_DATE, F_AF_LABEL, 'sku', F_AF_OFFTAKE, F_AF_PREDICTION_HORIZON]]
            df_fcst_eib.rename(columns={'sku': F_AF_SKU_WO_PKG}, inplace=True)
            df_fcst_eib[F_AF_SKU_WITH_PKG] = df_fcst_eib[F_AF_SKU_WO_PKG]
            df_fcst_eib[F_AF_CHANNEL] = 'NewChannel'  # no channel split for eib
            df_fcst_eib.replace({F_AF_LABEL: {'eib': V_EIB}}, inplace=True)

            return df_fcst_eib

        # prepare il history (offtake, sellin)
        def _prep_il_hist(_df_hist_il_channel, _df_hist_il_sellin, _df_il_split):
            # prepare IL sellin
            df_il_sellin_hist = _df_hist_il_sellin.loc[_df_hist_il_sellin['status'] == 'actual', :]
            df_il_sellin_hist = df_il_sellin_hist.loc[:, ['sku_no', 'date', 'volume']]
            df_il_sellin_hist = df_il_sellin_hist.groupby(['sku_no', 'date']).sum().reset_index()
            df_hist_sellin_split = pd.merge(df_il_sellin_hist,
                                            _df_il_split,
                                            how='left',
                                            left_on='sku_no',
                                            right_on='sku_code')
            df_hist_sellin_split[F_AF_SELLIN] = df_hist_sellin_split['volume'] * df_hist_sellin_split['split_ratio']
            df_hist_sellin_split = df_hist_sellin_split.loc[:, ['date', 'sku_code', 'channel', F_AF_SELLIN]]

            # merge offtake and sellin
            df_il_hist_channel = pd.merge(_df_hist_il_channel,
                                          df_hist_sellin_split,
                                          how='outer',
                                          on=['date', 'sku_code', 'channel'])
            df_il_hist_channel.rename(columns={'date': F_AF_DATE,
                                               'sku_code': F_AF_SKU_WO_PKG,
                                               'channel': F_AF_CHANNEL,
                                               'volume': F_AF_OFFTAKE}, inplace=True)

            # add additional fields
            df_il_hist_channel[F_AF_SKU_WITH_PKG] = df_il_hist_channel[F_AF_SKU_WO_PKG]
            df_il_hist_channel[F_AF_LABEL] = V_IL
            df_il_hist_channel[F_AF_PREDICTION_HORIZON] = V_ACTUAL

            return df_il_hist_channel

        # prepare di history (offtake, sellin, sellout)
        def _prep_di_hist(_df_hist_di, _df_di_pkg_split):
            # prepare offtake and sellout
            df_di_actual = _df_hist_di.loc[_df_hist_di['type'].isin(['offtake',
                                                                     'sellout',
                                                                     'sellin',
                                                                     'retailer_inv',
                                                                     'sp_inv'
                                                                     ]), :]
            df_di_actual = df_di_actual.loc[df_di_actual['status'] == 'actual', :]  # select historical actual only
            df_di_actual = df_di_actual[~((df_di_actual['date'] >= '2019-04-01') & (df_di_actual['sp'] == 'all'))]
            df_di_actual['quantity'] = df_di_actual['quantity'].fillna(0)  # fill null values with 0
            df_di_actual.rename(columns={'date': F_AF_DATE,
                                         'sku_wo_pkg': F_AF_SKU_WO_PKG,
                                         'sku_code': F_AF_SKU_WITH_PKG,
                                         'channel': F_AF_CHANNEL}, inplace=True)
            cols_di_index = [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_SKU_WITH_PKG, F_AF_CHANNEL]
            df_di_actual = df_di_actual.loc[:, cols_di_index + ['type', 'quantity']]

            # reshape offtake and sellout table
            df_off_so = df_di_actual.loc[df_di_actual['type'].isin(['offtake', 'sellout']), :]
            df_off_so = df_off_so.loc[df_off_so['channel'] != 'Total', :]  # exclude channel totals
            df_di_actual_pivot = df_off_so.pivot_table(index=cols_di_index, columns='type')  # unmelt table
            df_di_actual_pivot.columns = df_di_actual_pivot.columns.droplevel().rename(None)  # reformat table
            df_di_actual_pivot = df_di_actual_pivot.reset_index().fillna("null")

            # prepare sellin (no historical channel split)
            df_sellin_actual = df_di_actual.loc[df_di_actual['type'].isin(['sellin', 'retailer_inv', 'sp_inv']), :]
            df_sellin_actual = df_sellin_actual.loc[df_sellin_actual['channel'] == 'Total', :]
            df_sellin_actual = df_sellin_actual.pivot_table(index=cols_di_index, columns='type')  # unmelt table
            df_sellin_actual.columns = df_sellin_actual.columns.droplevel().rename(None)  # reformat table
            df_sellin_actual = df_sellin_actual.reset_index()
            df_sellin_actual = df_sellin_actual.loc[:,
                               [F_AF_DATE, F_AF_SKU_WITH_PKG, 'sellin', 'retailer_inv', 'sp_inv']]
            df_sellin_actual[['sellin', 'retailer_inv', 'sp_inv']].fillna(0, inplace=True)
            df_sellin_actual = df_sellin_actual.groupby([F_AF_DATE, F_AF_SKU_WITH_PKG]).sum().reset_index()
            df_sellin_split = pd.merge(df_sellin_actual,
                                       _df_di_pkg_split,
                                       how='right',
                                       left_on=F_AF_SKU_WITH_PKG,
                                       right_on='sku_code')
            df_sellin_split[F_AF_SELLIN] = df_sellin_split['sellin'] * df_sellin_split['split_ratio']
            df_sellin_split[F_AF_RETAILER_INV] = df_sellin_split['retailer_inv'] * df_sellin_split['split_ratio']
            df_sellin_split[F_AF_SP_INV] = df_sellin_split['sp_inv'] * df_sellin_split['split_ratio']
            df_sellin_split = df_sellin_split.loc[:, cols_di_index + [F_AF_SELLIN, F_AF_RETAILER_INV, F_AF_SP_INV]]

            # merge sellin with main table
            df_di_actual_all = pd.merge(df_di_actual_pivot,
                                        df_sellin_split,
                                        how='outer',
                                        on=cols_di_index)
            df_di_actual_all.fillna({F_AF_OFFTAKE: 0,
                                     F_AF_SELLOUT: 0,
                                     F_AF_SELLIN: 0,
                                     F_AF_RETAILER_INV: 0,
                                     F_AF_SP_INV: 0}, inplace=True)

            # add indicator columns
            df_di_actual_all[F_AF_LABEL] = V_DI
            df_di_actual_all[F_AF_PREDICTION_HORIZON] = V_ACTUAL

            return df_di_actual_all

        def _prep_eib_hist(_df_hist_eib, _df_hist_sellin):
            # prepare eib offtake
            df_eib_offtake = _df_hist_eib.loc[_df_hist_eib['scope'] == 'EIB', :]  # select offtake only
            df_eib_offtake.loc[:, 'date_match'] = pd.to_datetime(df_eib_offtake['produced_date']) - datetime.timedelta(
                days=60)
            df_eib_offtake = df_eib_offtake.loc[pd.to_datetime(df_eib_offtake['date']) < df_eib_offtake['date_match'],
                             :]
            df_eib_offtake.rename(columns={'volume': F_AF_OFFTAKE,
                                           'sku_code': F_AF_SKU_WO_PKG}, inplace=True)
            df_eib_offtake[F_AF_OFFTAKE] = df_eib_offtake[F_AF_OFFTAKE].fillna(0)  # fill null values with 0
            df_eib_offtake = df_eib_offtake.loc[:, [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_OFFTAKE]]
            df_eib_offtake = df_eib_offtake.groupby([F_AF_DATE, F_AF_SKU_WO_PKG]).sum().reset_index()

            # prepare eib sellin
            df_eib_sellin = _df_hist_sellin.loc[_df_hist_sellin['scope'] == 'EIB', :]
            df_eib_sellin = df_eib_sellin.loc[df_eib_sellin['status'] == 'actual', :]
            df_eib_sellin.rename(columns={'volume': F_AF_SELLIN,
                                          'sku_no': F_AF_SKU_WO_PKG}, inplace=True)
            df_eib_sellin = df_eib_sellin.loc[:, [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_SELLIN]]
            df_eib_sellin = df_eib_sellin.groupby([F_AF_DATE, F_AF_SKU_WO_PKG]).sum().reset_index()

            # merge offtake and sellin
            df_eib_hist = pd.merge(df_eib_offtake,
                                   df_eib_sellin,
                                   how='left',
                                   on=[F_AF_DATE, F_AF_SKU_WO_PKG])

            # add indicator columns
            df_eib_hist[F_AF_SKU_WITH_PKG] = df_eib_hist[F_AF_SKU_WO_PKG]
            df_eib_hist[F_AF_LABEL] = V_EIB
            df_eib_hist[F_AF_PREDICTION_HORIZON] = V_ACTUAL
            df_eib_hist[F_AF_CHANNEL] = 'NewChannel'

            return df_eib_hist

        # generate all fcst output
        df_fcst_split_di, df_di_pkg_split = _gran_di(df_hist_di, df_fcst_il_all)
        df_fcst_split_il, df_il_split, df_hist_il_channel = _gran_il(df_hist_il, df_fcst_il_all)
        df_fcst_eib = _prep_eib_fcst(df_fcst_il_all)

        # generate all actual output
        df_actual_il = _prep_il_hist(df_hist_il_channel, df_hist_il_sellin, df_il_split)
        df_actual_di = _prep_di_hist(df_hist_di, df_di_pkg_split)
        df_actual_eib = _prep_eib_hist(df_hist_eib, df_hist_il_sellin)

        # add other keys
        # get keys
        unique_key_df = df_mapping[['Country_acc', 'Brand_acc', 'Tier_acc', 'Stage_acc', 'SKU_wo_pkg']]\
            .drop_duplicates(inplace=False)
        unique_key_df.rename(columns={'Country_acc': F_AF_COUNTRY,
                                      'Brand_acc': F_AF_BRAND,
                                      'Tier_acc': F_AF_SUBBRAND,
                                      'Stage_acc': F_AF_STAGE,
                                      'SKU_wo_pkg': 'sku_code'}, inplace=True)

        # merge
        df_fcst_split_di = pd.merge(
            left=df_fcst_split_di,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )
        df_fcst_split_il = pd.merge(
            left=df_fcst_split_il,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )
        df_fcst_eib = pd.merge(
            left=df_fcst_eib,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )

        df_actual_il = pd.merge(
            left=df_actual_il,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )

        df_actual_di = pd.merge(
            left=df_actual_di,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )

        df_actual_eib = pd.merge(
            left=df_actual_eib,
            right=unique_key_df,
            left_on=F_AF_SKU_WO_PKG,
            right_on='sku_code'
        )

        # concatenate all output
        df_fcst_split_il_all = pd.concat([df_fcst_split_di, df_fcst_split_il, df_fcst_eib], sort=False)
        df_actual_all = pd.concat([df_actual_di,
                                   df_actual_eib,
                                   df_actual_il], sort=False)

        df_fcst_split_il_all.drop(columns=['sku_code'], inplace=True)
        df_actual_all.drop(columns=['sku_code'], inplace=True)

        # fill null values
        df_fcst_split_il_all.fillna({F_AF_OFFTAKE: 0}, inplace=True)
        df_actual_all.fillna({F_AF_SELLIN: 0,
                              F_AF_SELLOUT: 0,
                              F_AF_OFFTAKE: 0,
                              F_AF_RETAILER_INV: 0,
                              F_AF_SP_INV: 0
                              }, inplace=True)

        # update sku_with_pkg field
        df_fcst_split_il_all[F_AF_SKU_WITH_PKG] = df_fcst_split_il_all[F_AF_SKU_WITH_PKG] \
            .apply(lambda x: x.split('_')[-2] if len(x.split('_')) == 5 else 'N/A')
        df_actual_all[F_AF_SKU_WITH_PKG] = df_actual_all[F_AF_SKU_WITH_PKG] \
            .apply(lambda x: x.split('_')[-2] if len(x.split('_')) == 5 else 'N/A')

        # add the superlabel column
        df_fcst_split_il_all[F_AF_SUPERLABEL] = V_IL
        df_actual_all[F_AF_SUPERLABEL] = V_IL

        # export output
        df_fcst_split_il_all.fillna(value=0, inplace=True)
        df_actual_all.fillna(value=0, inplace=True)
        df_actual_all[F_AF_SELLOUT] = df_actual_all[F_AF_SELLOUT].replace(to_replace="null", value=0)

        df_actual_all.loc[:, [F_AF_SELLIN, F_AF_SELLOUT]] = 0

        return df_fcst_split_il_all, df_actual_all

    @staticmethod
    def add_granularity_dc(
            df_hist_dc: pd.DataFrame,
            df_fcst_dc: pd.DataFrame,
            df_hist_sellout: pd.DataFrame,
            df_hist_sellin: pd.DataFrame,
            df_prodlist: pd.DataFrame,
            df_custlist: pd.DataFrame,
            df_sp_inv: pd.DataFrame
    ):
        df_hist_dc = df_hist_dc.drop_duplicates()
        # map SKU code
        df_prodlist = df_prodlist.loc[:, ['sku_no', 'stage']].drop_duplicates()
        df_hist_dc = pd.merge(df_hist_dc, df_prodlist, how='left', left_on='sku_no', right_on='sku_no')
        df_hist_dc.rename(columns={'stage': F_AF_SKU_WO_PKG}, inplace=True)
        # map store/customer code
        df_custlist = df_custlist.loc[:, ['store_code', 'channel']]
        df_custlist['store_code'] = df_custlist['store_code'].str.lstrip('0')
        df_hist_dc['store_code'] = df_hist_dc['store_code'].astype(str)
        df_custlist[F_AF_CHANNEL] = df_custlist['channel'].replace({'MT': 'KA',
                                                                    'IMBS': 'GT',
                                                                    'KMBS': 'KA',
                                                                    'EC': 'EC',
                                                                    'Others': 'GT'})
        df_hist_dc = pd.merge(df_hist_dc, df_custlist, how='left', on='store_code')

        # select and prepare relevant historical data
        # df_hist_dc = df_hist_dc.loc[df_hist_dc['scope'] == 'DC', :]  # select DC offtake only
        df_hist_dc['quantity'] = df_hist_dc['quantity'].fillna(0)  # fill null values with 0
        df_hist_dc[F_AF_CHANNEL] = df_hist_dc[F_AF_CHANNEL].fillna('GT')
        df_actual_offtake = df_hist_dc.loc[:, [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL, 'quantity']]  # for export
        df_actual_offtake = df_actual_offtake.groupby([F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL]).sum().reset_index()
        df_actual_offtake.rename(columns={'quantity': F_AF_OFFTAKE}, inplace=True)
        df_actual_offtake[F_AF_DATE] = [pd.to_datetime(date).replace(day=1) for date in df_actual_offtake[F_AF_DATE]]
        df_hist_dc = df_hist_dc.loc[:, [F_AF_SKU_WO_PKG, 'channel', 'quantity']]  # select relevant columns
        df_hist_dc = df_hist_dc.groupby(
            [F_AF_SKU_WO_PKG, 'channel']).sum().reset_index()  # group by sku and channel (aggregate all dates)
        # aggregate to SKU total level
        df_hist_dc_sku_total = df_hist_dc.groupby([F_AF_SKU_WO_PKG]).sum().reset_index()  # obtain sku-level total
        df_hist_dc_sku_total.rename(columns={'quantity': 'sku_total_quantity'}, inplace=True)
        df_hist_dc_merged = pd.merge(df_hist_dc, df_hist_dc_sku_total, how='left',
                                     on=F_AF_SKU_WO_PKG)  # merge sku totals with granular quantity
        # compute split ratio
        df_hist_dc_merged['split_ratio'] = df_hist_dc_merged['quantity'] / df_hist_dc_merged[
            'sku_total_quantity']  # calculate ratio of sku total
        df_hist_dc_merged = df_hist_dc_merged.loc[:, [F_AF_SKU_WO_PKG, 'channel', 'split_ratio']]
        df_hist_dc_merged.to_csv(os.path.join(DIR_CACHE, 'dc_channel_split.csv'), index=False)
        # merge with fcst data
        df_fcst_dc.rename(columns={'yhat': F_AF_OFFTAKE}, inplace=True)
        df_fcst_dc[F_AF_LABEL] = V_DC
        df_fcst_dc = df_fcst_dc.loc[:, [F_AF_DATE, F_AF_LABEL, 'sku', F_AF_OFFTAKE, F_AF_PREDICTION_HORIZON]]
        # df_fcst_dc = df_fcst_dc.loc[df_fcst_dc[F_AF_PREDICTION_HORIZON] > 0, :]
        df_fcst_split_dc = pd.merge(df_fcst_dc, df_hist_dc_merged, how='right', left_on='sku',
                                    right_on=F_AF_SKU_WO_PKG)

        # apply split ratio to fcst
        df_fcst_split_dc['fcst vol'] = df_fcst_split_dc[F_AF_OFFTAKE] * df_fcst_split_dc['split_ratio']
        df_fcst_split_dc = df_fcst_split_dc.loc[:,
                           [F_AF_DATE, F_AF_LABEL, 'sku', F_AF_CHANNEL, 'fcst vol', F_AF_PREDICTION_HORIZON]]
        df_fcst_split_dc = df_fcst_split_dc.loc[pd.notnull(df_fcst_split_dc['fcst vol']),
                           :]  # exclude SKUs that are out of scope
        df_fcst_split_dc.rename(columns={'sku': F_AF_SKU_WO_PKG,
                                         'fcst vol': F_AF_OFFTAKE
                                         }, inplace=True)

        # prepare sellout actuals
        df_actual_sellout = df_hist_sellout.loc[:, [F_AF_DATE, 'sku_no', 'quantity']]
        df_actual_sellout[F_AF_DATE] = [pd.to_datetime(date).replace(day=1) for date in df_actual_sellout[F_AF_DATE]]
        df_actual_sellout = df_actual_sellout.groupby([F_AF_DATE, 'sku_no']).sum().reset_index()
        df_actual_sellout = pd.merge(df_actual_sellout, df_prodlist, how='left', left_on='sku_no', right_on='sku_no')
        df_actual_sellout_split = pd.merge(df_actual_sellout,
                                           df_hist_dc_merged,
                                           how='right',
                                           left_on='stage',
                                           right_on=F_AF_SKU_WO_PKG)
        df_actual_sellout_split['act vol'] = df_actual_sellout_split['quantity'] * df_actual_sellout_split[
            'split_ratio']
        df_actual_sellout_split = df_actual_sellout_split.loc[:, [F_AF_DATE, 'stage', F_AF_CHANNEL, 'act vol']]
        df_actual_sellout_split = df_actual_sellout_split.groupby(
            [F_AF_DATE, 'stage', F_AF_CHANNEL]).sum().reset_index()
        df_actual_sellout_split.rename(columns={'act vol': F_AF_SELLOUT,
                                                'stage': F_AF_SKU_WO_PKG}, inplace=True)

        # prepare sellin actuals
        df_hist_sellin.rename(columns={'sku': 'stage',
                                       'sellin_dc': 'quantity'}, inplace=True)
        df_actual_sellin = df_hist_sellin.loc[:, [F_AF_DATE, 'stage', 'quantity']]
        df_actual_sellin[F_AF_DATE] = [pd.to_datetime(date).replace(day=1) for date in df_actual_sellin[F_AF_DATE]]
        df_actual_sellin = df_actual_sellin.groupby([F_AF_DATE, 'stage']).sum().reset_index()
        df_actual_sellin_split = pd.merge(df_actual_sellin,
                                          df_hist_dc_merged,
                                          how='right',
                                          left_on='stage',
                                          right_on=F_AF_SKU_WO_PKG)
        df_actual_sellin_split['act vol'] = df_actual_sellin_split['quantity'] * df_actual_sellin_split['split_ratio']
        df_actual_sellin_split = df_actual_sellin_split.loc[:, [F_AF_DATE, 'stage', F_AF_CHANNEL, 'act vol']]
        df_actual_sellin_split = df_actual_sellin_split.groupby(
            [F_AF_DATE, 'stage', F_AF_CHANNEL]).sum().reset_index()
        df_actual_sellin_split.rename(columns={'act vol': F_AF_SELLIN,
                                               'stage': F_AF_SKU_WO_PKG}, inplace=True)

        # prepare sp inv actuals
        df_sp_inv_hist = df_sp_inv.loc[:, [F_AF_DATE, 'sku', 'quantity']]
        # df_sp_inv_hist = df_sp_inv_hist.loc[df_sp_inv_hist['sku'].isin(SELECTED_SKUS), :]
        df_sp_inv_hist[F_AF_DATE] = [pd.to_datetime(date).replace(day=1) for date in df_sp_inv_hist[F_AF_DATE]]
        df_sp_inv_hist = df_sp_inv_hist.groupby([F_AF_DATE, 'sku']).sum().reset_index()
        df_sp_inv_hist_split = pd.merge(df_sp_inv_hist,
                                        df_hist_dc_merged,
                                        how='right',
                                        left_on='sku',
                                        right_on=F_AF_SKU_WO_PKG)
        df_sp_inv_hist_split['inv vol'] = df_sp_inv_hist_split['quantity'] * df_sp_inv_hist_split['split_ratio']
        df_sp_inv_hist_split = df_sp_inv_hist_split.loc[:, [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL, 'inv vol']]
        df_sp_inv_hist_split = df_sp_inv_hist_split.groupby(
            [F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL]).sum().reset_index()
        df_sp_inv_hist_split.rename(columns={'inv vol': F_AF_SP_INV}, inplace=True)

        # merge all actual data
        df_actual_all = pd.merge(df_actual_offtake,
                                 df_actual_sellout_split,
                                 how='outer',
                                 on=[F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL])
        df_actual_all = pd.merge(df_actual_all,
                                 df_actual_sellin_split,
                                 how='outer',
                                 on=[F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL])
        df_actual_all = pd.merge(df_actual_all,
                                 df_sp_inv_hist_split,
                                 how='outer',
                                 on=[F_AF_DATE, F_AF_SKU_WO_PKG, F_AF_CHANNEL])
        df_actual_all.fillna({F_AF_OFFTAKE: 0,
                              F_AF_SELLIN: 0,
                              F_AF_SELLOUT: 0,
                              F_AF_SP_INV: 0}, inplace=True)

        # add additional fields for fcst
        df_fcst_split_dc[F_AF_DATE] = pd.to_datetime(df_fcst_split_dc[F_AF_DATE])
        df_fcst_split_dc.replace({F_AF_LABEL: {'dc': V_DC}}, inplace=True)
        df_fcst_split_dc[F_AF_COUNTRY] = V_CN
        df_fcst_split_dc[F_AF_SKU_WITH_PKG] = df_fcst_split_dc[F_AF_SKU_WO_PKG]
        df_fcst_split_dc[F_AF_BRAND] = df_fcst_split_dc[F_AF_SKU_WO_PKG].astype(str).str[:2]
        df_fcst_split_dc[F_AF_SUBBRAND] = df_fcst_split_dc[F_AF_BRAND]
        df_fcst_split_dc[F_AF_STAGE] = df_fcst_split_dc[F_AF_SKU_WO_PKG].astype(str).str[2:]

        # add additional fields for actual
        df_actual_all[F_AF_DATE] = pd.to_datetime(df_actual_all[F_AF_DATE])  # , format='%m/%d/%Y')
        df_actual_all[F_AF_LABEL] = V_DC
        df_actual_all[F_AF_COUNTRY] = V_CN
        df_actual_all[F_AF_PREDICTION_HORIZON] = V_ACTUAL
        df_actual_all[F_AF_SKU_WITH_PKG] = df_actual_all[F_AF_SKU_WO_PKG]
        df_actual_all[F_AF_BRAND] = df_actual_all[F_AF_SKU_WO_PKG].astype(str).str[:2]
        df_actual_all[F_AF_SUBBRAND] = df_actual_all[F_AF_BRAND]
        df_actual_all[F_AF_STAGE] = df_actual_all[F_AF_SKU_WO_PKG].astype(str).str[2:]
        df_actual_all = df_actual_all.loc[df_actual_all[F_AF_SKU_WO_PKG].isin(SELECTED_SKUS_DC)].copy()

        # add the superlabel column
        df_fcst_split_dc[F_AF_SUPERLABEL] = V_DC
        df_actual_all[F_AF_SUPERLABEL] = V_DC

        df_actual_all.loc[:, [F_AF_SELLIN, F_AF_SELLOUT]] = 0

        return df_fcst_split_dc, df_actual_all

    @staticmethod
    def detect_dups(df: pd.DataFrame, group_cols: List[str] = [F_AF_DATE] + V_AF_LOWEST_GRANULARITY) -> None:
        count_dups = df.groupby(by=group_cols).agg({
            col: 'count' for col in df.columns if col not in group_cols
        }).reset_index(inplace=False)

        to_show_dups = np.any([count_dups[col] > 1 for col in count_dups.columns if col not in group_cols], axis=0)

        dups = df.loc[to_show_dups]

        # we cannot calculate tradeflow if for one item, one date is associated to many values
        if len(dups.index) > 0:
            raise ValueError(f'Forecast data contains duplicates: \n{dups.head().transpose()}')
