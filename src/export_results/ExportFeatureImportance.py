# coding: utf-8
import logging
from datetime import datetime

import pandas as pd
from dateutil.relativedelta import relativedelta

from src.scenario import *
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

logger = logging.getLogger(__name__)


class ExportFeatureImportance:

    def __init__(self, feature_importance_df: pd.DataFrame, cycle_date: datetime, umb_label: str):
        self.feature_importance_df = feature_importance_df.rename(columns={
            'index': 'feature'
        }, inplace=False)
        self.feature_importance_df.fillna(value=0, inplace=True)
        self.feature_importance_df[F_DN_CYC_DAT] = cycle_date + relativedelta(
            months=-self.feature_importance_df['horizon'].min()
        )
        self.feature_importance_df[F_DN_LV2_UMB_BRD_COD] = umb_label
        self.feature_importance_df['horizon'] = self.feature_importance_df['horizon'].apply(lambda x: int(x))

    def to_impala(self, impala_connect_cfg, to_database: str, to_table: str) -> None:
        """ Exports the feature importance to Impala

        :param impala_connect_cfg: Impala connection configuration
        :param to_database: Target database
        :param to_table: Target table
        :return:
        """
        cyc_dat = self.feature_importance_df[F_DN_CYC_DAT].iloc[0]
        label = self.feature_importance_df[F_DN_LV2_UMB_BRD_COD].iloc[0]

        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_connect_cfg)
        impala_cursor = impala_connection.cursor()

        logger.info(f'Clearing `{to_database}`.`{to_table}` to insert new values for cycle...')

        sql_delete_req_0 = f"""
            DROP TABLE IF EXISTS `{to_database}`.`_{to_table}`;
        """
        sql_delete_req_1 = f"""
            CREATE TABLE `{to_database}`.`_{to_table}` AS SELECT * FROM `{to_database}`.`{to_table}` 
            WHERE NOT(`{F_DN_CYC_DAT}`='{cyc_dat}' AND `{F_DN_LV2_UMB_BRD_COD}`='{label}');
        """
        sql_delete_req_2 = f"""
            DROP TABLE `{to_database}`.`{to_table}`;
        """
        sql_delete_req_3 = f"""
            CREATE TABLE `{to_database}`.`{to_table}` AS SELECT * FROM `{to_database}`.`_{to_table}`;
        """
        sql_delete_req_4 = f"""
            DROP TABLE `{to_database}`.`_{to_table}`;
        """
        sql_reqs = [sql_delete_req_0, sql_delete_req_1, sql_delete_req_2, sql_delete_req_3, sql_delete_req_4]
        for req in sql_reqs:
            logger.debug(SQLUtils.escape_req(req[:1000]) + ' [...]')
            impala_cursor.execute(req)

        sql_insert_req = SQLUtils.make_insert_req_from_dataframe_infer_types(
            to_database=to_database,
            to_table=to_table,
            df=self.feature_importance_df
        )
        logger.debug(SQLUtils.escape_req(sql_insert_req[:1000]) + ' [...]')
        impala_cursor.execute(sql_insert_req)
        impala_connection.close()
