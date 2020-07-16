# coding: utf-8
import logging
import os

import pandas as pd
import yaml

from cfg.paths import DIR_CFG
from src.data_wrangling import F_VERSION, F_INDEX, F_TABLE_NAME, F_TABLES, F_DATABASE, F_DATA, F_CONNECT
from src.scenario import V_IL
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

F_DI_TRADEFLOW = 'DI_TRADEFLOW'
F_FILE_DESC = 't_src_fil_dsc'
F_SOURCE_TIME = 't_rec_src_tst'
F_INSERT_TIME = 't_rec_ins_tst'

logger = logging.getLogger(__name__)


def get_latest_di_tradeflow() -> pd.DataFrame:
    """ This functions gets the latest DI Tradeflow data stored in Impala. It has been implemented in case EIB sellin has to
    be updated over more than the latest months.
    :return:
    """
    with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
        cfg = yaml.load(ymlf)

    impala_connection = ImpalaUtils.get_impala_connector(cfg=cfg[F_CONNECT])

    database_name = cfg[F_DATA][V_IL][F_DATABASE]
    table_name = cfg[F_DATA][V_IL][F_TABLES][F_DI_TRADEFLOW][F_TABLE_NAME]
    index_cols = cfg[F_DATA][V_IL][F_TABLES][F_DI_TRADEFLOW][F_INDEX]
    version_col = cfg[F_DATA][V_IL][F_TABLES][F_DI_TRADEFLOW][F_VERSION]

    # selects the latest version of data
    logger.info(f'Reading {database_name}.{table_name}...')

    req = SQLUtils.get_fetch_latest(database_name, index_cols, table_name, version_col)
    logger.debug('Request: %s' % SQLUtils.escape_req(req))
    di_tradeflow_df = pd.read_sql(sql=req, con=impala_connection)

    cols_drop = [F_FILE_DESC, F_SOURCE_TIME, F_INSERT_TIME]
    di_tradeflow_df.drop(cols_drop, axis=1, inplace=True, errors='ignore')

    return di_tradeflow_df


if __name__ == '__main__':  # Quick testing
    df = get_latest_di_tradeflow()
    print('OK')
