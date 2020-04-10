import re
import time
from datetime import datetime
from typing import List
from typing import Union, Dict, Any

import pandas as pd

from src.scenario import F_DN_DATE_FMT

V_VERY_SMALL = 1e-6

F_SQL_VALUES = 'sql_values'


class SQLUtils:
    @staticmethod
    def get_fetch_latest(database_name: str, index_cols: List[str], table_name: str, version_col: str,
                         where_clause: str = '') -> str:
        """ Generates an Impala query to get the latest data from a table.

        :param database_name: Target database
        :param index_cols: Column names acting as key for group by
        :param table_name: Target table name
        :param version_col: Column acting as version column
        :param where_clause: Additional where clause
        :return: Impala query
        """
        req = f"""
                    SELECT content.* FROM `{database_name}`.`{table_name}` content
                        INNER JOIN (
                            SELECT {','.join([f'`{table_name}`.`{col_name}`' for col_name in index_cols])}, 
                            MAX(`{version_col}`) `{version_col}`
                            FROM `{database_name}`.`{table_name}` {where_clause} 
                            GROUP BY {','.join([f'`{table_name}`.`{col_name}`' for col_name in index_cols])}
                        ) selector
                        ON {' AND '.join([f'content.`{col_name}` = selector.`{col_name}`' for col_name in index_cols])}
                            AND content.`{version_col}` = selector.`{version_col}`
                """
        return req

    # DATE SKU OFFTAKE INS_TST
    # X    X    X        2019-07-07 00:00:00  <--
    # X    X    Y        2019-07-06 00:00:00

    # DATE SKU  INS_TST
    # X    X    2019-07-07 00:00:00

    @staticmethod
    def make_insert_req_from_series(to_database: str, to_table: str, df: pd.Series) -> str:

        def to_str(val: Union[datetime, str, float, bool, int]) -> str:
            if isinstance(val, datetime):
                return f"'{val.strftime(F_DN_DATE_FMT)}'"
            elif isinstance(val, float):
                return f"{val:.6f}"[:19]
            elif isinstance(val, int) or isinstance(val, bool):
                return str(val)
            elif isinstance(val, str):
                return f"'{val}'"

        values = ','.join([to_str(v) for v in df.values])
        columns = ','.join(df.index.values)
        req = f"INSERT INTO `{to_database}`.`{to_table}` ({columns}) VALUES ({values})"

        return req

    @staticmethod
    def make_insert_req_from_dataframe_infer_types(to_database: str, to_table: str, df: pd.DataFrame) -> str:
        if df.empty:
            raise ValueError("Data is empty")

        def to_str(val: Union[datetime, str, float, bool, int]) -> str:
            if isinstance(val, datetime):
                return f"'{val.strftime(F_DN_DATE_FMT)}'"
            elif isinstance(val, float):
                val = 0 if -V_VERY_SMALL < val < 0 else val
                return f"{val:.6f}"[:19]
            elif isinstance(val, int) or isinstance(val, bool):
                return str(val)
            elif isinstance(val, str):
                return f"'{val}'"

        values = ', '.join(df.apply(lambda row: '(' + ', '.join([to_str(v) for v in row.values]) + ')', axis=1))
        columns = ','.join(df.columns.values)
        req = f"INSERT INTO `{to_database}`.`{to_table}` ({columns}) VALUES {values}"
        if 'null' in req:
            print(' NO ')
        return req

    @staticmethod
    def make_insert_req_from_dataframe_dtypes(
            to_database: str,
            to_table: str,
            df: pd.DataFrame,
            col_dtypes: Dict[str, Any]
    ) -> str:
        if df.empty:
            raise ValueError("Data is empty")
        if F_SQL_VALUES in df.columns:
            raise ValueError(f'{F_SQL_VALUES} is a reserved column name')

        sql_df = pd.DataFrame()
        for col, dtype in col_dtypes.items():
            if dtype == str:
                sql_df[col] = df[col].apply(lambda x: f"'{x}'")
            elif dtype in {int, bool}:
                sql_df[col] = df[col].apply(lambda x: str(x))
            elif dtype == float:
                sql_df[col] = df[col].apply(lambda x: f"{x:.6f}"[:19])
            elif dtype == datetime:
                sql_df[col] = df[col].apply(lambda x: f"'{x.strftime(F_DN_DATE_FMT)}'")

        sql_df[F_SQL_VALUES] = sql_df.apply(lambda x: ','.join(x), axis=1)

        values = '(' + '),('.join(sql_df[F_SQL_VALUES]) + ')'
        columns = ','.join(df.columns.values)
        req = f"INSERT INTO `{to_database}`.`{to_table}` ({columns}) VALUES {values}"

        return req

    @staticmethod
    def escape_req(req: str) -> str:
        return re.sub(' +', ' ', req.replace("\n", " "))


def quick_test():
    to_database = 'dummy_db'
    to_table = 'dummy_table'
    n = 10000
    df = pd.DataFrame({
        'a': [3] * n,
        'b': [True] * n,
        'c': [5.39] * n,
        'd': ['yes'] * n,
    })
    col_dtypes = {'a': int, 'b': bool, 'c': float, 'd': str}

    start = time.time()
    req = SQLUtils.make_insert_req_from_dataframe_dtypes(to_database=to_database, to_table=to_table, df=df,
                                                         col_dtypes=col_dtypes)
    end = time.time()
    print(end - start)
    print(req)

    start = time.time()
    req = SQLUtils.make_insert_req_from_dataframe_infer_types(to_database=to_database, to_table=to_table, df=df)
    end = time.time()
    print(end - start)
    print(req)


if __name__ == '__main__':
    quick_test()
