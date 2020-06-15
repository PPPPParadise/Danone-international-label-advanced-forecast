# coding: utf-8
"""
This module contains the RawMasterDC class which inherits from the RawMaster base class. The RawMasterDC class is required to
construct the raw_master table for the DC demand forecast models.
"""
import logging
import os

import numpy as np
import pandas as pd

from cfg.paths import DIR_CACHE
from src.data_wrangling import F_DC_POS_PRODUCT_LIST, F_DC_POS, F_DC_OSA, F_DC_ANP, F_DC_SELLIN, F_DC_SPINV, \
    F_DC_SELLOUT, F_DC_STORE_DISTRIB, F_DC_SELLIN_HIST, SELECTED_SKUS_DC
from src.data_wrangling.RawMaster import RawMaster
from src.utils.misc import set_date_to_first_of_month

logger = logging.getLogger(__name__)

FROM_CACHE = False


class RawMasterDC(RawMaster):
    """This class contains functions to create the DC raw_master table"""

    def load_dc_pos_product_mapping(self) -> pd.DataFrame:
        """ Loads and prepares DC POS product mapping that maps other identifiers (e.g. sku_no) to sku

        :return: pandas dataframe containing product mapping
        """
        logger.info('loading and preparing DC POS product mapping')
        dc_pos_product_mapping = self._raw_data[F_DC_POS_PRODUCT_LIST]
        dc_pos_product_mapping.columns = dc_pos_product_mapping.columns.str.lower()
        dc_pos_product_mapping = dc_pos_product_mapping.drop_duplicates()

        dc_pos_product_mapping.rename(columns={'code': 'sku_no'}, inplace=True)
        for col in ['sku_no', 'brand', 'stage']:
            dc_pos_product_mapping[col] = dc_pos_product_mapping[col].astype(str).str.strip()

        dc_pos_product_mapping = dc_pos_product_mapping.rename(columns={'stage': 'sku'})
        dc_pos_product_mapping = dc_pos_product_mapping.loc[dc_pos_product_mapping.sku.isin(SELECTED_SKUS_DC), :]

        # dc_pos_product_mapping.loc[dc_pos_product_mapping['sku'].str.contains('AP'), 'brand'] = 'AP'
        dc_pos_product_mapping['brand'] = dc_pos_product_mapping['sku'].str[:2]

        assert dc_pos_product_mapping.groupby('sku')['brand'].nunique().max() == 1, 'only one brand per sku'

        return dc_pos_product_mapping[['sku_no', 'sku', 'brand']].drop_duplicates()

    def load_dc_offtake_data(self) -> pd.DataFrame:
        """ Loads and prepares DC offtake data

        :return: Pandas dataframe containing prepared DC offtake data
        """
        logger.info('loading and preparing DC offtake data')

        offtake = self._raw_data[F_DC_POS]
        product_mapping = self.load_dc_pos_product_mapping()

        offtake.columns = offtake.columns.str.lower()
        offtake = offtake.drop_duplicates()  # avoid string mismatch in Impala query
        offtake['date'] = pd.to_datetime(offtake['date'])
        offtake['date'] = set_date_to_first_of_month(offtake.date)

        mapped_offtake = pd.merge(left=offtake[['date', 'sku_no', 'quantity']],
                                  right=product_mapping[['sku_no', 'sku', 'brand']],
                                  on='sku_no',
                                  how='inner',
                                  validate='many_to_one',
                                  suffixes=(False, False))

        mapped_offtake = mapped_offtake.drop(['sku_no'], axis=1)
        rename_dict = {'quantity': 'offtake_dc'}
        mapped_offtake = mapped_offtake.rename(columns=rename_dict)
        mapped_offtake = mapped_offtake.loc[mapped_offtake.sku.isin(SELECTED_SKUS_DC), :].copy(deep=True).reset_index(
            drop=True)

        grouped_offtake = mapped_offtake[['date', 'sku', 'brand', 'offtake_dc']].groupby(
            ['date', 'sku', 'brand'], as_index=False
        ).sum()

        return grouped_offtake

    def load_dc_osa_data(self) -> pd.DataFrame:
        """ Loads and prepares DC OSA data

        :return: Pandas dataframe containing DC OSA data
        """
        logger.info('loading and preparing DC OSA data')

        dc_osa = self._raw_data[F_DC_OSA]
        dc_osa.columns = dc_osa.columns.str.lower()
        dc_osa['month_conv'] = dc_osa['month'].astype(str)
        dc_osa['month_conv'] = dc_osa['month_conv'].apply(lambda x: '0'+x if x == '1' else x)
        dc_osa['date'] = dc_osa['year'].astype(str) + dc_osa['month_conv'].astype(str) + '01'
        dc_osa['date'] = pd.to_datetime(dc_osa['date'], format='%Y%m%d')
        dc_osa = dc_osa.drop(['year', 'month'], axis=1)
        return dc_osa

    def load_dc_anp_spending_data(self) -> pd.DataFrame:
        """ Loads and prepares DC ANP spending data

        :return: Pandas dataframe containing prepared DC ANP spending data
        """
        logger.info('loading and preparing DC ANP data')

        dc_anp = self._raw_data[F_DC_ANP]
        dc_anp.columns = dc_anp.columns.str.lower()

        dc_anp['date'] = pd.to_datetime(dc_anp['date'])
        dc_anp['date'] = set_date_to_first_of_month(dc_anp.date)

        dc_anp['brand'] = dc_anp['brand'].replace(to_replace={'C&G': 'CG', 'Karicare': 'KG'})

        dc_anp = dc_anp.rename(columns={'spending': 'anp'})

        return dc_anp

    def load_dc_sellin_data(self) -> pd.DataFrame:
        """ Loads and prepares DC sellin data

        :return: Pandas dataframe contianing prepared DC sellin data
        """
        logger.info('loading and preparing DC sellin data')

        sellin = self._raw_data[F_DC_SELLIN]
        sellin_hist = self._raw_data[F_DC_SELLIN_HIST]

        # prepare cycle sellin
        sellin.rename(columns={'bil_dat': 'date',
                               'bil_doc_typ_cod': 'order_type',
                               'mat_cod': 'sku_no',
                               'sal_cus_cod': 'sp_code',
                               'bil_sku_qty': 'quantity',
                               'sku_uom_cod': 'unit'}, inplace=True)

        sellin = sellin[sellin['sal_org_cod'].astype(str) == '7850'].copy()  # code 7850 stands for DC

        sellin['date'] = pd.to_datetime(sellin['date'])
        sellin['date'] = set_date_to_first_of_month(sellin.date)
        sellin = sellin.loc[sellin['date'] >= '2019-01-01', :]

        # calibrate return orders
        list_return_type = ['ZRE', 'ZS1']  # return orders to be deducted
        sellin['adj'] = np.where(sellin['order_type'].isin(list_return_type),
                                 sellin['quantity'] * (-1),
                                 sellin['quantity'])

        sellin = sellin.loc[:, ['date', 'sku_no', 'quantity']]

        # prepare history sellin
        sellin_hist = sellin_hist.loc[:, ['date', 'sku_no', 'quantity']]
        sellin_hist['date'] = set_date_to_first_of_month(sellin_hist.date)

        # concatenate history and cycle file
        sellin_all = pd.concat([sellin, sellin_hist])
        sellin_all['sku_no'] = sellin_all['sku_no'].astype(str)

        # apply mapping
        product_mapping = self.load_dc_pos_product_mapping()
        mapped_sellin = pd.merge(left=sellin_all[['date', 'sku_no', 'quantity']],
                                 right=product_mapping[['sku_no', 'sku', 'brand']],
                                 on='sku_no',
                                 how='inner',
                                 validate='many_to_one',
                                 suffixes=(False, False))
        mapped_sellin = mapped_sellin.drop(['sku_no'], axis=1)
        mapped_sellin = mapped_sellin.rename(columns={'quantity': 'sellin_dc'})

        mapped_sellin = mapped_sellin.loc[mapped_sellin.sku.isin(SELECTED_SKUS_DC), :]

        grouped_sellin = mapped_sellin[
            ['date', 'sku', 'brand', 'sellin_dc']
        ].groupby(['date', 'sku', 'brand'], as_index=False).sum()

        grouped_sellin = grouped_sellin.loc[grouped_sellin.sku.isin(SELECTED_SKUS_DC), :].copy(deep=True).reset_index(
            drop=True)
        grouped_sellin = grouped_sellin.loc[:, ['date', 'sku', 'brand', 'sellin_dc']]
        grouped_sellin.to_pickle(os.path.join(DIR_CACHE, 'dc_sellin.pkl'))

        return grouped_sellin[['date', 'sku', 'brand', 'sellin_dc']]

    def load_dc_sp_inv_data(self) -> pd.DataFrame:
        """ Load and prepare DC SP inventory data

        :return: Pandas dataframe containing prepared DC SP inventory data
        """
        logger.info('loading and preparing DC SP INV data')

        sp_inv = self._raw_data[F_DC_SPINV]
        sp_inv.columns = sp_inv.columns.str.lower()
        sp_inv.rename(columns={'date_time': 'date'}, inplace=True)
        sp_inv['date'] = pd.to_datetime(sp_inv['date'])
        sp_inv['date'] = set_date_to_first_of_month(sp_inv.date)

        sp_inv = sp_inv.loc[sp_inv.sku.isin(SELECTED_SKUS_DC), :].copy()
        sp_inv['brand'] = sp_inv['sku'].str[:2]
        sp_inv = sp_inv.rename(columns={'quantity': 'sp_inv_dc'})  # rename columns to correct brand abbreviations
        sp_inv['sku'] = sp_inv['sku'].replace({'AP1/380': 'AP1MINI'})

        sp_inv_agg = sp_inv[['date', 'sku', 'brand', 'sp_inv_dc']].\
            groupby(['date', 'sku', 'brand'], as_index=False).agg({'sp_inv_dc': 'sum'})
        return sp_inv_agg

    def load_dc_sellout_data(self) -> pd.DataFrame:
        """ Load and prepare DC sellout data

        :return: Pandas dataframe containing prepared DC sellout data
        """
        logger.info('loading and preparing DC sellout data')

        sellout = self._raw_data[F_DC_SELLOUT]
        sellout.columns = sellout.columns.str.lower()

        sellout['date'] = pd.to_datetime(sellout['date'])
        sellout['date'] = set_date_to_first_of_month(sellout.date)

        # mapping
        product_mapping = self.load_dc_pos_product_mapping()
        mapped_sellout = pd.merge(left=sellout[['date', 'sku_no', 'quantity', 'revenue', 'price']],
                                  right=product_mapping[['sku_no', 'sku', 'brand']],
                                  on='sku_no',
                                  how='inner',
                                  validate='many_to_one',
                                  suffixes=(False, False))

        mapped_sellout = mapped_sellout.rename(columns={'quantity': 'sellout', 'price': 'sellout_price', 'revenue': 'sellout_revenue'})

        agg_dict = {'sellout': 'sum',
                    'sellout_revenue': 'sum',
                    'sellout_price': 'mean'}
        agg_sellout = mapped_sellout.groupby(['date', 'sku', 'brand'], as_index=False).agg(agg_dict)

        filtered_sellout = agg_sellout.loc[agg_sellout.sku.isin(SELECTED_SKUS_DC), :]

        return filtered_sellout

    def load_dc_store_distribution_data(self) -> pd.DataFrame:

        logger.info('loading and preparing DC store distribution data')

        dc_store_distrib = self._raw_data[F_DC_STORE_DISTRIB]

        dc_store_distrib['date'] = pd.to_datetime(
            pd.DataFrame({'Year': dc_store_distrib['month'].astype(str).str[:4],
                          'Month': dc_store_distrib['month'].astype(str).str[-2:], 'Day': 1}), errors='coerce')
        dc_store_distrib = dc_store_distrib.drop(['month'], axis=1)
        dc_store_distrib = pd.melt(dc_store_distrib, id_vars='date',
                                   value_vars=['ap', 'ac', 'nc'],
                                   var_name='brand', value_name='store_distribution')
        dc_store_distrib['brand'] = dc_store_distrib['brand'].str.upper()
        return dc_store_distrib

    def make_raw_master(self) -> pd.DataFrame:
        """ Reads individual data sources and assembles the raw_master_DC by joining them consecutively.

        :return: Pandas dataframe containing assembled raw_master_DC
        """
        logger.info('assembling RawMasterDC')
        dc_offtake = self.load_dc_offtake_data()
        dc_osa = self.load_dc_osa_data()
        dc_anp = self.load_dc_anp_spending_data()
        dc_sellin = self.load_dc_sellin_data()
        dc_spinv = self.load_dc_sp_inv_data()
        dc_sellout = self.load_dc_sellout_data()
        dc_store_distrib = self.load_dc_store_distribution_data()

        # initialize master with offtake data
        raw_master = dc_offtake

        # join dc_osa data
        merge_cols = ['date', 'sku']
        cols_to_add = ['osa', 'csl']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_osa[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # join dc_anp spending data
        merge_cols = ['date', 'brand']
        cols_to_add = ['anp']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_anp[merge_cols + cols_to_add],
            on=merge_cols,
            how='left',
            validate='many_to_one',
            suffixes=(False, False)
        )

        # join dc_sellin data
        merge_cols = ['date', 'sku']
        cols_to_add = ['sellin_dc']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_sellin[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # join dc_spinv data
        merge_cols = ['date', 'sku', 'brand']
        cols_to_add = ['sp_inv_dc']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_spinv[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # join dc_sellout data
        merge_cols = ['date', 'sku', 'brand']
        cols_to_add = ['sellout', 'sellout_revenue', 'sellout_price']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_sellout[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='one_to_one',
            suffixes=(False, False)
        )

        # join dc_store_distribution data
        merge_cols = ['date', 'brand']
        cols_to_add = ['store_distribution']
        raw_master = pd.merge(
            left=raw_master,
            right=dc_store_distrib[merge_cols + cols_to_add],
            on=merge_cols,
            how='outer',
            validate='many_to_one',
            suffixes=(False, False)
        )

        raw_master = raw_master.sort_values(['date', 'sku'])
        raw_master = raw_master.loc[pd.notnull(raw_master['offtake_dc']), :]

        return raw_master
