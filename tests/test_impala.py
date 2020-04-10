# coding: utf-8
import logging
import os
import traceback
import unittest

import pandas as pd
import yaml
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from impala.dbapi import connect

from cfg.paths import DIR_DATA
from cfg.paths import LOG_FILENAME, DIR_CFG, DIR_TEST_OUTPUT
from src.data_wrangling import F_VERSION, F_INDEX, F_TABLE_NAME, F_TABLES, \
    F_DATABASE, F_DATA, F_CONNECT
from src.data_wrangling.Data import Data
from src.scenario import *
from src.utils.impala_utils import ImpalaUtils
from src.utils.logging import setup_loggers
from src.utils.sql_utils import SQLUtils

logger = logging.getLogger(__name__)
setup_loggers('test_impala', log_filepath=LOG_FILENAME, level=logging.INFO)


class TestImpalaConnection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Load configuration
        """
        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            cfg = yaml.load(ymlf)
        cls.conn = ImpalaUtils.get_impala_connector(cfg=cfg[F_CONNECT])
        cls.database_name = "s_edl_rtc_stg"
        cls.table_names = [
            'src_chi_frc_category',
            'src_chi_frc_customer',
            'src_chi_frc_dc_anp',
            'src_chi_frc_dc_osa',
            'src_chi_frc_di_tradeflow',
            'src_chi_frc_distributor',
            'src_chi_frc_dms_sales',
            'src_chi_frc_eib_osa',
            'src_chi_frc_eib_price',
            'src_chi_frc_il_competitor',
            'src_chi_frc_il_offtake',
            'src_chi_frc_il_offtake_all',
            'src_chi_frc_il_offtake_channel',
            'src_chi_frc_il_sellin',
            'src_chi_frc_offtake_u1',
            'src_chi_frc_offtake_yuou',
            'src_chi_frc_pos',
            'src_chi_frc_product_list',
            'src_chi_frc_sap_cus_cod',
            'src_chi_frc_sap_mat_cod',
            'src_chi_f_snd_bil_doc',
            'src_chi_frc_sellout_u1',
            'src_chi_frc_spinv',
            'src_chi_frc_stage_sku_split',
            'src_chi_frc_store_dist',
            'src_chi_frc_sellin_his'
        ]

    @unittest.skip('Skipped')
    def test_dump_impala(self):

        for table_name in self.table_names:
            req = f"""
                SELECT * FROM `{self.database_name}`.`{table_name}`;
            """
            try:
                df = pd.read_sql(sql=req, con=self.conn)
                df.to_csv(
                    os.path.join(DIR_TEST_OUTPUT, f'{table_name}.csv')
                )
                logger.info(f'[OK] {table_name}')
            except Exception as e:
                logger.error(f'[Execution failed] {table_name}. {e}')
                logger.error(f'[Traceback] {traceback.format_exc()}')

    @unittest.skip('Skipped')
    def test_dump_tinton(self):
        req = f"""
            SELECT * FROM `s_edl_rtc_stg`.`src_chi_frc_tin2ton`;
        """
        df = pd.read_sql(sql=req, con=self.conn)
        df.to_csv(
            os.path.join(DIR_DATA, f'src_chi_frc_sku_tin2ton.csv')
        )
        logger.info(f'[OK] src_chi_frc_sku_tin2ton')

    @unittest.skip('Skipped')
    def test_dump_latest_impala(self):
        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            cfg = yaml.load(ymlf)

        for label in Data.KNOWN_LABELS:
            database_name, table_cfg = cfg[F_DATA][label][F_DATABASE], cfg[F_DATA][label][F_TABLES]
            for key, entry in table_cfg.items():
                table_name = entry[F_TABLE_NAME]
                index_cols = entry[F_INDEX]
                version_col = entry[F_VERSION]

                req = SQLUtils.get_fetch_latest(
                    database_name=database_name, index_cols=index_cols, table_name=table_name, version_col=version_col
                )
                try:
                    df = pd.read_sql(sql=req, con=self.conn)
                    df.to_csv(
                        os.path.join(DIR_TEST_OUTPUT, f'latest_{table_name}.csv')
                    )
                    logger.info(f'[OK] {table_name}')
                except Exception as e:
                    logger.error(f'[Execution failed] {table_name}. {e}')
                    logger.error(f'[Traceback] {traceback.format_exc()}')

    @unittest.skip('Skipped')
    def test_explore_impala(self):

        for table_name in self.table_names:
            logger.info(f'Getting count and first rows of {table_name}')
            try:
                req = f"""
                    SELECT count(*) FROM `{self.database_name}`.`{table_name}`;
                """
                df = pd.read_sql(sql=req, con=self.conn)
                logger.info(df.head().transpose())

                req = f"""
                    SELECT * FROM `{self.database_name}`.`{table_name}` LIMIT 10;
                """
                df = pd.read_sql(sql=req, con=self.conn)
                logger.info(df.head().transpose())
            except Exception as e:
                logger.error(f'[Execution failed] {table_name}. {e}')
                logger.error(f'[Traceback] {traceback.format_exc()}')

    # @unittest.skip('Skipped')
    def test_explore_f_dmd_frc_unified_tinton(self):
        req = f"""
            SELECT DISTINCT frc_usr_nam_dsc, frc_cre_dat, frc_mdf_dat, lv2_umb_brd_cod, lv3_pdt_brd_cod
             FROM `s_cn3_rtc_dwh`.`f_dmd_frc_unified_tinton`
             ORDER BY frc_mdf_dat;
        """
        df = pd.read_sql(sql=req, con=self.conn)
        self.assertTrue(len(df.index) > 0)
        logger.info(f'[OK] src_chi_frc_sku_tin2ton')
