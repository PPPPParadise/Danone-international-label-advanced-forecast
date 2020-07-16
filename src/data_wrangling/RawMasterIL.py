# coding: utf-8
"""
This module contains the RawMasterIL class which inherits from the RawMaster base class. The RawMasterIL class is required to
construct the raw_master table for the IL/EIB/DI demand forecast models.
"""
import logging

import numpy as np
import pandas as pd

from src.data_wrangling import F_CATEGORY_FORECAST, F_SMARTPATH, F_EIB_PRICE, F_EIB_OSA, F_IL_SELLIN, \
    F_MAPPING_SKU_STD_INFO, F_MAPPING_OSA_EIB, F_IL_OFFTAKE, F_DI_TRADEFLOW, SELECTED_SKUS_IL
from src.data_wrangling.RawMaster import RawMaster
from src.utils.misc import set_date_to_first_of_month

logger = logging.getLogger()


class RawMasterIL(RawMaster):
    """This class contains functions to create the IL raw_master table"""

    def make_raw_master(self) -> pd.DataFrame:
        """ Reads and assembles the data sources.

        :return: Data frame containing the raw data
        """
        logger.info('assembling RawMasterIL')
        di_tradeflow = self.load_di_tradeflow_data()
        sellin = self.load_il_sellin_data()
        offtake = self.load_il_offtake_data()
        eib_osa = self.load_eib_osa_data()
        eib_price = self.load_eib_price_data()
        smartpath = self.load_smartpath_competitor_data()
        category_ts = self.load_category_forecast()

        logger.info('Assembling Raw Master...')
        prepared_offtake_data = self.prepare_offtake_data(offtake)
        prepared_sellin_data = self.prepare_sellin_data(sellin)
        prepared_tradeflow_data = self.prepare_tradeflow_data(tradeflow=di_tradeflow)

        # initialize master with offtake data
        raw_master = prepared_offtake_data

        # join sellin data
        merge_cols = ['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f', 'date']
        cols_to_add = ['sellin_eib']  # , 'sellin_di']
        raw_master = pd.merge(
            left=raw_master,
            right=prepared_sellin_data[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # join tradeflow_data
        merge_cols = ['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f', 'date']
        cols_to_add = ['retailer_inv', 'sellout', 'sp_inv', 'sellin_di']
        # ,'sellin_di'
        for col in ['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f']:
            raw_master[col] = raw_master[col].astype(str)
            prepared_tradeflow_data[col] = prepared_tradeflow_data[col].astype(str)
        raw_master = pd.merge(
            left=raw_master,
            right=prepared_tradeflow_data[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # merge osa data
        merge_cols = ['country', 'brand', 'tier', 'stage_3f', 'date']
        cols_to_add = ['isa', 'osa']
        raw_master = pd.merge(
            left=raw_master,
            right=eib_osa[merge_cols + cols_to_add],
            on=merge_cols,
            how='left',
            validate='many_to_one',
            suffixes=(False, False)
        )

        # merge price data
        merge_cols = ['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'date']
        cols_to_add = ['price']
        eib_price = eib_price.drop_duplicates(merge_cols)
        raw_master = pd.merge(
            left=raw_master,
            right=eib_price[merge_cols + cols_to_add],
            on=merge_cols,
            how='left',
            validate='many_to_one',
            suffixes=(False, False)
        )

        # merge smartpath data
        merge_cols = ['date']
        cols_to_add = ['price_krmb_per_ton_il', 'value_krmb_il', 'volume_ton_il']
        raw_master = pd.merge(
            left=raw_master,
            right=smartpath[merge_cols + cols_to_add],
            on=merge_cols,
            how='left',
            validate='many_to_one',
            suffixes=(False, False)
        )

        # add category forecast and babypool data
        merge_cols = ['date']
        cols_to_add = ['total_vol', 'if_vol', 'fo_vol', 'gum_vol', 'cl_vol', 'il_vol', '0to6_month_population',
                       '6to12_month_population',
                       '12to36_month_population']
        raw_master = pd.merge(left=raw_master,
                              right=category_ts[merge_cols + cols_to_add],
                              on=merge_cols,
                              validate='many_to_one',
                              how='left',
                              suffixes=(False, False)
                              )

        raw_master = self.create_il_columns(raw_master)

        # raw_master = raw_master.fillna(0)
        raw_master['date'] = pd.to_datetime(raw_master['date'])  # this should already be the case!

        # exclude SKUs not in scope
        raw_master = raw_master.loc[raw_master['sku_wo_pkg'].isin(SELECTED_SKUS_IL), :]

        return raw_master

    def load_mapping_sku_std_info(self) -> pd.DataFrame:
        """ Load standard SKU mapping table for IL

        :return:
        """
        sku_mapping = self._raw_data[F_MAPPING_SKU_STD_INFO]
        sku_mapping.columns = sku_mapping.columns.str.lower()

        sku_mapping = sku_mapping.rename(columns={'country_acc': 'country', 'brand_acc': 'brand', 'tier_acc': 'tier', 'stage_acc': 'stage'})

        for col in sku_mapping.columns:
            sku_mapping[col] = sku_mapping[col].apply(str)
            sku_mapping[col] = sku_mapping[col].str.strip()
        return sku_mapping

    def load_mapping_osa_eib(self) -> pd.DataFrame:
        """ Load OSA EIB mapping table

        :return:
        """
        sku_mapping = self._raw_data[F_MAPPING_OSA_EIB]

        # for col in sku_mapping.columns:
        #    sku_mapping[col] = sku_mapping[col].str.strip()

        return sku_mapping

    def create_il_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Construct offtake_il and sellin_il by adding eib and DI values

        :param df: Pandas dataframe containing offtake/sellin data for EIB and DI
        :return: Original pandas input dataframe with added IL columns for offtake and sellin
        """
        # calculate IL offtake by the sum of DI_offtake and EIB_offtake
        df['offtake_il'] = df['offtake_di'].fillna(0) + df['offtake_eib'].fillna(0)
        df['sellin_il'] = df['sellin_di'].fillna(0) + df['sellin_eib'].fillna(0)

        return df

    def load_di_tradeflow_data(self) -> pd.DataFrame:
        """ Loads DI_TRADEFLOW data

        :return: Pandas dataframe holding di_tradeflow_data
        """
        logger.info('loading DI tradeflow data...')
        di_trade_flow_data = self._raw_data[F_DI_TRADEFLOW].rename(columns={'sku_code': 'sku_std'})

        # correct legacy typo
        di_trade_flow_data['status'] = di_trade_flow_data['status'].replace({'acutual': 'actual'})

        di_trade_flow_data['date'] = pd.to_datetime(arg=di_trade_flow_data['date'],
                                                    format='%m/%d/%Y',
                                                    errors='raise')
        di_trade_flow_data['date'] = set_date_to_first_of_month(di_trade_flow_data['date'])

        sku_mapping = self.load_mapping_sku_std_info()
        di_trade_flow_data = pd.merge(left=di_trade_flow_data,
                                      right=sku_mapping[['sku_std',
                                                         # 'sku_wo_pkg',
                                                         'stage_3f']],
                                      on='sku_std',
                                      how='inner',
                                      validate='many_to_one')

        return di_trade_flow_data

    def load_il_sellin_data(self) -> pd.DataFrame:
        """ Loads IL sellin data

        :return: il_sellin data
        """
        logger.info('Loading IL Sellin data...')
        sellin = self._raw_data[F_IL_SELLIN].copy()
        sellin = sellin.drop_duplicates()
        sellin['date'] = pd.to_datetime(sellin['date'])
        sellin['date'] = set_date_to_first_of_month(sellin['date'])
        # sellin['produced_date'] = pd.to_datetime(sellin['produced_date'])
        sellin.rename(columns={'volume': 'sellin', 'sku_no': 'sku_wo_pkg'}, inplace=True)
        # merge_cols = ['scope', 'date', 'produced_date', 'sku_wo_pkg']
        merge_cols = ['scope', 'date', 'sku_wo_pkg']
        # sellin_agg = sellin.groupby(merge_cols, as_index=False).agg({'sellin': 'sum', 'weight_per_tin': 'mean'})
        sellin_agg = sellin.groupby(merge_cols, as_index=False).agg({'sellin': 'sum'})

        sku_mapping = self.load_mapping_sku_std_info()
        sellin_agg = pd.merge(left=sellin_agg,
                              right=sku_mapping
                              [['sku_wo_pkg', 'country', 'brand', 'stage', 'tier', 'stage_3f']].drop_duplicates(),
                              on='sku_wo_pkg',
                              how='inner', validate='many_to_one')
        return sellin_agg

    def load_il_offtake_data(self) -> pd.DataFrame:
        """ Loads IL_OFFTAKE data

        :return: il_offtake data
        """
        logger.info('Loading IL Offtake data...')
        offtake = self._raw_data[F_IL_OFFTAKE]

        offtake.rename(columns={'date_time': 'date'}, inplace=True)
        offtake['date'] = pd.to_datetime(offtake['date'])
        offtake['date'] = set_date_to_first_of_month(offtake['date'])

        offtake['produced_date'] = pd.to_datetime(offtake['produced_date'])
        offtake.rename(columns={'volume': 'offtake', 'sku_code': 'sku_wo_pkg'}, inplace=True)
        merge_cols = ['scope', 'date', 'produced_date', 'sku_wo_pkg']
        offtake_agg = offtake.groupby(merge_cols, as_index=False).agg({'offtake': 'sum'})

        # correct legacy sku mapping
        offtake_agg['sku_wo_pkg'] = offtake_agg['sku_wo_pkg'].replace({'ANZ_KC_COW_1': 'ANZ_KC_GD_1',
                                                                       'ANZ_KC_COW_2': 'ANZ_KC_GD_2',
                                                                       'ANZ_KC_COW_3': 'ANZ_KC_GD_3',
                                                                       'ANZ_KC_COW_4': 'ANZ_KC_GD_4',
                                                                       'ANZ_KC_GOAT_1': 'ANZ_KC_GT_1',
                                                                       'ANZ_KC_GOAT_2': 'ANZ_KC_GT_2',
                                                                       'ANZ_KC_GOAT_3': 'ANZ_KC_GT_3'})

        sku_mapping = self.load_mapping_sku_std_info()
        offtake_agg = pd.merge(left=offtake_agg,
                               right=sku_mapping
                               [['sku_wo_pkg', 'country', 'brand', 'stage', 'tier', 'stage_3f']].drop_duplicates(),
                               on='sku_wo_pkg',
                               how='inner', validate='many_to_one')

        return offtake_agg

    def load_eib_osa_data(self) -> pd.DataFrame:
        """ Loads eib_osa and eib_isa data.

        :return: pandas dataframe holding eib_osa data
        """
        logger.info('Loading EIB OSA data...')
        eib_osa = self._raw_data[F_EIB_OSA]

        # cast string to datetime format
        eib_osa['date'] = pd.to_datetime(arg=eib_osa['month'],
                                         format='%Y-%m',
                                         errors='raise')

        # merge with sku mapping
        eib_osa.rename(columns={'item': 'Item'}, inplace=True)
        eib_osa_sku_mapping = self.load_mapping_osa_eib()
        eib_osa_mapped = pd.merge(left=eib_osa,
                                  right=eib_osa_sku_mapping[
                                      ['Item', 'Country', 'Brand', 'Tier', 'Stage_3F', 'OSA_ISA']],
                                  on='Item',
                                  how='inner',
                                  validate='many_to_one')

        eib_osa_pivoted = eib_osa_mapped.pivot_table(index=['Stage_3F', 'Tier', 'Brand', 'Country', 'date'],
                                                     columns='OSA_ISA',
                                                     values='value', aggfunc=np.mean).reset_index()

        eib_osa_pivoted.columns = pd.Series(eib_osa_pivoted.columns).str.lower()
        eib_osa_pivoted['osa'] = eib_osa_pivoted['osa'].fillna(value=-1)
        eib_osa_pivoted['isa'] = eib_osa_pivoted['isa'].fillna(value=-1)

        # clean whitespace at the beginning and end of granularity columns
        for col in ['stage_3f', 'tier', 'brand', 'country']:
            eib_osa_pivoted[col] = eib_osa_pivoted[col].str.strip()

        return eib_osa_pivoted

    def load_eib_price_data(self) -> pd.DataFrame:
        """ Loads EIB_PRICE data

        :return: pandas dataframe holding eib_price data
        """
        logger.info('Loading EIB Price data...')

        eib_price = self._raw_data[F_EIB_PRICE]

        # cast string to datetime format
        eib_price['date'] = pd.to_datetime(arg=eib_price['date'],
                                           # format='%Y-%m-%d',
                                           format='%m/%d/%Y',
                                           errors='raise')

        eib_price['date'] = set_date_to_first_of_month(pd.to_datetime(eib_price['date']))

        eib_price.rename(columns={'sub_brand': 'tier', 'sku_code': 'sku_wo_pkg'}, inplace=True)

        # clean whitespace at the beginning and end of granularity columns
        for col in ['sku_wo_pkg', 'sku', 'country', 'brand', 'tier', 'stage', 'source', 'scope']:
            eib_price[col] = eib_price[col].apply(str)
            eib_price[col] = eib_price[col].str.strip()

        eib_price = eib_price.groupby(['sku_wo_pkg', 'date', 'sku', 'country', 'brand', 'tier', 'stage', 'source']).\
            mean().reset_index()

        return eib_price

    def load_smartpath_competitor_data(self) -> pd.DataFrame:
        """ Load Smart path competitor volumes and values

        :return: Pandas dataframe holding smartpath competitor data
        """
        logger.info('Loading SmartPath competitors data...')
        smartpath = self._raw_data[F_SMARTPATH]
        smartpath.rename(columns={'volume': 'volume_ton',
                                  'value': 'value_kRMB',
                                  'scope': 'Scope',
                                  'price': 'price_kRMB_per_ton'}, inplace=True)
        smartpath = pd.pivot_table(smartpath,
                                   values=['volume_ton', 'value_kRMB', 'price_kRMB_per_ton'],
                                   index=['date'], columns=['Scope'],
                                   aggfunc={'volume_ton': 'sum', 'value_kRMB': 'sum', 'price_kRMB_per_ton': 'mean'},
                                   fill_value=0)
        smartpath.columns = ['_'.join(col).strip() for col in smartpath.columns]
        smartpath = smartpath.reset_index()
        smartpath['date'] = set_date_to_first_of_month(pd.to_datetime(smartpath['date']))
        smartpath = smartpath.loc[smartpath['date'] >= '2017-01-01', :]
        smartpath.columns = [col.lower() for col in smartpath.columns]

        return smartpath

    def load_category_forecast(self) -> pd.DataFrame:
        """ Loads category forecast inlcuding babypool information

        :return: pandas dataframe holding category forecast
        """
        logger.info('Loading Category forecast data...')
        category_ts = self._raw_data[F_CATEGORY_FORECAST]
        category_ts.rename(columns={'month': 'Month'}, inplace=True)
        category_ts['date'] = pd.to_datetime(pd.DataFrame({
            'Year': category_ts.Month.str.slice(0, 4).astype(int),
            'Month': category_ts.Month.str.slice(5, 7).astype(int),
            'Day': 1
        }))

        category_ts['date'] = set_date_to_first_of_month(category_ts['date'])

        category_ts.columns = [col.replace(' ', '_').replace('-', 'to').lower() for col in category_ts.columns]
        category_ts.rename(columns={'0_6_month_population': '0to6_month_population',
                                    '6_12_month_population': '6to12_month_population',
                                    '12_36_month_population': '12to36_month_population'}, inplace=True)

        '''
        category_ts_long = pd.melt(category_ts,
                                   id_vars='date',
                                   value_vars=['IF vol', 'FO vol', 'GUM vol'],
                                   var_name='stage_3f', value_name='category_by_stage')
    
        category_ts_long['stage_3f'] = category_ts_long['stage_3f'].replace(
            {'IF vol': 'IF', 'FO vol': 'FO', 'GUM vol': 'GUM'})
        '''

        return category_ts  # , category_ts_long

    @staticmethod
    def prepare_tradeflow_data(tradeflow: pd.DataFrame) -> pd.DataFrame:
        """ Pivots tradeflow data, treat channel == sp and others separately.
        Extracts basic features (e.g.: sp_inv, retailer_inv, ...)

        :param tradeflow: Pandas dataframe holding tradeflow data
        :return: Pandas dataframe holding prepared tradeflow data
        """

        tradeflow['channel'] = tradeflow['channel'].replace({'all': 'Total'})
        tradeflow_total = tradeflow.loc[tradeflow.channel == 'Total', :]
        tradeflow_channels = tradeflow.loc[tradeflow.channel != 'Total', :]

        di_tradeflow_channels_pivot = pd.pivot_table(data=tradeflow_channels,
                                                     values='quantity',
                                                     index=['sku_wo_pkg', 'country', 'brand', 'tier', 'stage',
                                                            'stage_3f', 'date'],
                                                     columns='type',
                                                     aggfunc='sum',
                                                     fill_value=0)[
            ['offtake', 'sellout']]

        di_tradeflow_total_pivot = pd.pivot_table(data=tradeflow_total,
                                                  values='quantity',
                                                  index=['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f',
                                                         'date'],
                                                  columns='type',
                                                  aggfunc='sum',
                                                  fill_value=0)[
            ['sellin', 'retailer_inv', 'sp_inv']]

        di_tradeflow_pivot = pd.merge(
            left=di_tradeflow_total_pivot,
            right=di_tradeflow_channels_pivot,
            how='outer',
            on=['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f', 'date'],
            validate='one_to_one'
        )

        di_tradeflow_pivot = di_tradeflow_pivot.rename(columns={
            'offtake': 'offtake_tradeflow',
            'retailer_inv': 'retailer_inv',
            # 'retailer_inv_month': 'retailer_inv_month',
            'sellout': 'sellout',
            'sp_inv': 'sp_inv',
            # 'sp_inv_month': 'sp_inv_month',
            'sellin': 'sellin_di',
        })

        # we obtain offtake from the offtake file
        di_tradeflow_pivot = di_tradeflow_pivot.drop('offtake_tradeflow', axis=1)
        di_tradeflow_pivot = di_tradeflow_pivot.reset_index()

        return di_tradeflow_pivot

    @staticmethod
    def prepare_offtake_data(offtake: pd.DataFrame) -> pd.DataFrame:
        """ Calculates offtake pivot

        :param offtake: Offtake data frame
        :return: Pivoted Offtake
        """
        offtake_pivot = pd.pivot_table(data=offtake,
                                       values=['offtake'],
                                       index=['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f', 'date'],
                                       columns='scope',
                                       aggfunc='mean',
                                       fill_value=0)

        offtake_pivot.columns = ['_'.join(col).strip().lower().replace(" ", "_") for col in offtake_pivot.columns]
        offtake_pivot = offtake_pivot.reset_index()

        return offtake_pivot

    @staticmethod
    def prepare_sellin_data(sellin: pd.DataFrame) -> pd.DataFrame:
        """ Calculates Sell-in pivot

        :param sellin: Sell-in data frame
        :return: Pivoted Sell-in
        """
        sellin_pivot = pd.pivot_table(data=sellin,
                                      values=['sellin'],
                                      index=['sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'stage_3f', 'date'],
                                      columns='scope',
                                      aggfunc='mean',
                                      fill_value=0)

        sellin_pivot.columns = ['_'.join(col).strip().lower().replace(" ", "_") for col in sellin_pivot.columns]
        sellin_pivot = sellin_pivot.reset_index()

        # sellin_di from this source is not reliable
        # sellin_pivot = sellin_pivot.drop('sellin_di', axis=1)

        return sellin_pivot
