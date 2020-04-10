# coding: utf-8
import logging
import os
import pickle
from typing import Dict
from typing import List

import pandas as pd

from cfg.paths import DIR_CACHE, DIR_MAPPINGS, DIR_DATA, RUN_DATETIME
from src.data_wrangling import F_VERSION, F_INDEX, F_TABLE_NAME, F_TABLES, \
    F_DATABASE, F_DATA, F_CONNECT
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

F_DC_POS = 'DC_POS'
F_DC_SELLOUT = 'DC_SELLOUT'
F_DC_SELLIN = 'DC_SELLIN'
F_DC_SPINV = 'DC_SPINV'
F_DC_ANP = 'DC_ANP'
F_DC_OSA = 'DC_OSA'
F_DC_STORE_DISTRIB = 'DC_STORE_DISTRIB'
F_DC_CUSTOMER = 'dc_customer_list'
F_DC_POS_PRODUCT_LIST = 'DC_PRODUCT_LIST'
F_DC_DISTRIBUTOR = 'dc_distributor_list'
F_DC_MAP_LEGACY_SKU = 'dc_mapping_legacy_sku'
F_DC_MAP_LEGACY_SP = 'dc_mapping_legacy_sp'
F_IL_OFFTAKE = 'IL_OFFTAKE'
F_IL_SELLIN = 'IL_SELLIN'
F_DI_TRADEFLOW = 'DI_TRADEFLOW'
F_SMARTPATH = 'SMARTPATH'
F_IL_EIB_PRICE = 'EIB_PRICE'
F_EIB_OSA = 'EIB_OSA'
F_CATEGORY_FORECAST = 'CATEGORY_FORECAST'
F_IL_MAP_SKU_CN = 'il_mapping_sku_cn'
F_MAPPING_SKU_STD_INFO = 'MAPPING_SKU_STD_INFO'
F_IL_MAP_SKU_TREE = 'il_mapping_sku_tree'
F_IL_MAP_DI = 'il_mapping_di'
F_IL_MAP_SELLIN_EIB = 'il_mapping_sellin_eib'
F_MAPPING_OSA_EIB = 'MAPPING_OSA_EIB'
F_IL_MAP_OSW_ANZ = 'il_mapping_osw_anz'
F_IL_MAP_OSW_DE = 'il_mapping_osw_de'
F_IL_MAP_OSW_NL = 'il_mapping_osw_nl'
F_IL_OG = 'il_og'
F_IL_UPLIFT = 'il_uplift'

F_FILE_DESC = 't_src_fil_dsc'
F_SOURCE_TIME = 't_rec_src_tst'
F_INSERT_TIME = 't_rec_ins_tst'

logger = logging.getLogger(__name__)


class Data:
    F_IL = 'IL'
    F_DC = 'DC'
    KNOWN_LABELS = {F_DC, F_IL}

    # Constructors
    def __init__(self, data_dict: Dict[str, pd.DataFrame], source):
        self._data_dict = data_dict
        self.source = source

    @staticmethod
    def from_impala(label: str, cfg: dict) -> 'Data':
        """ Instanciates a Data object from the data available on Impala

        :param label: allowed values: 'DC', 'IL'
        :param cfg: Impala connection configuration
        :return: Data object
        """

        if label not in Data.KNOWN_LABELS:
            raise KeyError(f'Provided label {label} unknown. Known labels are {Data.KNOWN_LABELS}')

        impala_connection = ImpalaUtils.get_impala_connector(cfg=cfg[F_CONNECT])

        database_name, table_cfg = cfg[F_DATA][label][F_DATABASE], cfg[F_DATA][label][F_TABLES]
        data_dict = dict()
        for key, entry in table_cfg.items():
            table_name = entry[F_TABLE_NAME]
            index_cols = entry[F_INDEX]
            version_col = entry[F_VERSION]

            # selects the latest version of data
            logger.info(f'Reading {database_name}.{table_name}...')
            if table_name != 'src_chi_frc_product_list':
                req = SQLUtils.get_fetch_latest(database_name, index_cols, table_name, version_col)
                logger.debug('Request: %s' % SQLUtils.escape_req(req))
                df = pd.read_sql(sql=req, con=impala_connection)
            else:
                req = f"""
                      SELECT * FROM `{database_name}`.`{table_name}`
                """
                logger.debug('Request: %s' % SQLUtils.escape_req(req))
                df = pd.read_sql(sql=req, con=impala_connection)

            cols_drop = [F_FILE_DESC, F_SOURCE_TIME, F_INSERT_TIME]
            df.drop(cols_drop, axis=1, inplace=True, errors='ignore')

            # if table_name == 'src_chi_frc_il_sellin':
            #     df = df.groupby(by=['sku_no', 'scope', 'type', 'date', 'produced_date', 'status', 'unit']).agg({
            #         'volume': [('sum', 'sum'), ('count', 'count')]
            #     }).reset_index()
            #     df.columns = ['_'.join(col).strip() if 'volume' in col else ''.join(col).strip()
            #     for col in df.columns]
            #     df.loc[df['volume_count'] > 1, 'volume_sum'] \
            #         = df.loc[df['volume_count'] > 1, 'volume_sum'] / df.loc[df['volume_count'] > 1, 'volume_count']
            #     df.rename(columns={'volume_sum': 'volume'}, inplace=True)
            #     df = df[[c for c in df.columns if c != 'volume_count']]

            if table_name == 'src_chi_frc_di_tradeflow':
                df.replace({'channel': {'ALI': 'Ali', 'New channel': 'NewChannel'}}, inplace=True)
                df = df.groupby([c for c in df.columns if c != 'quantity']).sum().reset_index()

            # for col in cols_drop:
            #     if col in df.columns:
            #         df.drop(col, axis=1, inplace=True)

            data_dict[key] = df

            # Uncomment to read table samples easily
            # req = f""" SELECT * FROM {database_name}.{table_name} LIMIT 1 """
            # print(database_name, table_name)
            # print(pd.read_sql(sql=req, con=impala_connection).transpose())
            # print(' ')

        impala_connection.close()

        data_dict[F_MAPPING_SKU_STD_INFO] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'SKU_std_info.csv'))
        data_dict[F_MAPPING_OSA_EIB] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'F_OSA_EIB.csv'))
        data_dict[F_IL_MAP_SELLIN_EIB] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'M_IL_Sellin_EIB_DI.csv'))

        return Data(data_dict=data_dict, source='impala')

    @staticmethod
    def get_fetch_latest(database_name: str, index_cols: List[str], table_name: str, version_col: str) -> str:
        """ Makes the Impala query to fetch the latest data

        :param database_name: Database name
        :param index_cols: Columns names acting as key columns to group by
        :param table_name: Name of the table
        :param version_col: Name of the column defining the data version
        :return: Impala query to execute
        """
        req = f"""
                SELECT content.* FROM `{database_name}`.`{table_name}` content
                    INNER JOIN (
                        SELECT {','.join(
            [f'`{table_name}`.`{col_name}`' for col_name in index_cols])}, MAX(`{version_col}`) `{version_col}`
                        FROM `{database_name}`.`{table_name}`
                        GROUP BY {','.join([f'`{table_name}`.`{col_name}`' for col_name in index_cols])}
                    ) selector
                    ON {' AND '.join([f'content.`{col_name}` = selector.`{col_name}`' for col_name in index_cols])}
                        AND content.`{version_col}` = selector.`{version_col}`
            """
        logger.debug(SQLUtils.escape_req(req))
        return req

    @staticmethod
    def from_files(label: str) -> 'Data':
        """ This method is obsolete.

        :param label:
        :return:
        """
        if label not in Data.KNOWN_LABELS:
            raise KeyError(f'Provided label {label} unknown. Knowm labels are {Data.KNOWN_LABELS}')

        data_dict = Dict[str, pd.DataFrame]
        if label == 'DC':
            # content files
            data_dict[F_DC_POS] = pd.read_csv(os.path.join(DIR_DATA, 'DC_POS.csv'), sep=';')
            data_dict[F_DC_SELLOUT] = pd.read_csv(os.path.join(DIR_DATA, 'DC_sellout.csv'), sep=';')
            data_dict[F_DC_SELLIN] = pd.read_csv(os.path.join(DIR_DATA, 'DC_sellin.csv'), sep=';')
            data_dict[F_DC_SPINV] = pd.read_csv(os.path.join(DIR_DATA, 'DC_SPinv.csv'), sep=';')
            data_dict[F_DC_ANP] = pd.read_csv(os.path.join(DIR_DATA, 'DC_AnP.csv'), sep=';')
            data_dict[F_DC_OSA] = pd.read_csv(os.path.join(DIR_DATA, 'DC_OSA.csv'), sep=';')
            data_dict[F_DC_STORE_DISTRIB] = pd.read_csv(os.path.join(DIR_DATA, 'DC_store_distribution.csv'), sep=';')
            data_dict[F_DC_CUSTOMER] = pd.read_csv(os.path.join(DIR_DATA, 'CustomerList.csv'), sep=';')
            data_dict[F_DC_POS_PRODUCT_LIST] = pd.read_csv(os.path.join(DIR_DATA, 'ProductList.csv'), sep=';')
            data_dict[F_DC_DISTRIBUTOR] = pd.read_csv(os.path.join(DIR_DATA, 'Distributor.csv'), sep=';')
            # mapping files
            data_dict[F_DC_MAP_LEGACY_SKU] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'old_SKU_code_mapping_prep.csv'))
            data_dict[F_DC_MAP_LEGACY_SP] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'old_sp_code_mapping_prep.csv'))
        elif label == 'IL':
            # content files
            data_dict[F_IL_OFFTAKE] = pd.read_csv(os.path.join(DIR_DATA, 'IL_offtake_all_hist.csv'), sep=';')
            data_dict[F_IL_SELLIN] = pd.read_csv(os.path.join(DIR_DATA, 'IL_sellin.csv'), sep=';')
            data_dict[F_DI_TRADEFLOW] = pd.read_csv(os.path.join(DIR_DATA, 'DI_TRADEFLOW.csv'), sep=';')
            data_dict[F_SMARTPATH] = pd.read_csv(os.path.join(DIR_DATA, 'IL_smartpath_competitor.csv'), sep=';')
            data_dict[F_IL_EIB_PRICE] = pd.read_csv(os.path.join(DIR_DATA, 'EIB_PRICE.csv'), sep=';')
            data_dict[F_EIB_OSA] = pd.read_csv(os.path.join(DIR_DATA, 'EIB_OSA.csv'), sep=';')
            data_dict[F_CATEGORY_FORECAST] = pd.read_csv(os.path.join(DIR_DATA, 'input_ts.csv'), sep=';')
            # mapping files
            data_dict[F_IL_MAP_SKU_CN] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'SKU_list_w_CN.csv'))
            data_dict[F_MAPPING_SKU_STD_INFO] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'SKU_std_info.csv'))
            data_dict[F_IL_MAP_SKU_TREE] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'SKU_tree_Look_up.csv'))
            data_dict[F_IL_MAP_DI] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'M_DI.csv'))
            data_dict[F_IL_MAP_SELLIN_EIB] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'M_IL_Sellin_EIB_DI.csv'))
            data_dict[F_MAPPING_OSA_EIB] = pd.read_csv(os.path.join(DIR_MAPPINGS, 'F_OSA_EIB.csv'))
            data_dict[F_IL_MAP_OSW_ANZ] = pd.read_csv(
                os.path.join(DIR_MAPPINGS, 'IL_Automation_RowMapping_OSW_ANZ.csv'))
            data_dict[F_IL_MAP_OSW_DE] = pd.read_csv(
                os.path.join(DIR_MAPPINGS, 'IL_Automation_RowMapping_OSW_Ger.csv'))
            data_dict[F_IL_MAP_OSW_NL] = pd.read_csv(
                os.path.join(DIR_MAPPINGS, 'IL_Automation_RowMapping_OSW_NL.csv'))

        return Data(data_dict=data_dict, source='files')

    # caching
    @staticmethod
    def from_pickle(filename: str) -> 'Data':
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Loading {__class__} from pickle {fpath}...')
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, filename: str) -> None:
        """ Exports self to pickle in cache.

        :param filename: Target filename
        :return:
        """
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Saving {__class__} to pickle {fpath}...')
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    def to_csv(self, path: str) -> None:
        """ Exports the data contained in self to CSV

        :param path: Target path
        :return:
        """
        logger.info(f'Saving {__class__} to {path}...')
        for k, d in self._data_dict.items():
            try:
                d.to_csv(os.path.join(path, f"{RUN_DATETIME.strftime('%Y%m%d%H%M%S')}_processauto_{k}.csv"))
            except:
                logging.warning(f'Could not save {k} to {path}')

    # Accessor
    def __getitem__(self, item_id: str):
        if item_id not in self._data_dict:
            raise KeyError(f'Key "{item_id}" available. Available: {str(self._data_dict)}.')
        return self._data_dict[item_id]
