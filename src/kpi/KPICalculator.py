# coding: utf-8
import datetime
import logging

import dateutil
import numpy as np
import pandas as pd

import src.kpi.kpis_computation as kpis
from src.kpi import INDEX_COLS_ACTUALS, INDEX_COLS_FORECASTS, F_DN_ETL_TST, F_DN_MEA_DAT, F_DN_APO_FLG, \
    F_DN_FRC_MDF_DAT, F_DN_FRC_MTH_NBR, F_DN_FRC_FLG
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

logger = logging.getLogger(__name__)


class KPICalculator:

    def __init__(self, forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame):
        self.variable = ['oft_trk_val', 'sal_ins_val']
        self.rel_col = ['cyc_dat', 'lv2_umb_brd_cod', 'lv3_pdt_brd_cod', 'mat_cod', 'mea_dat', 'dis_chl_cod'] + self.variable
        self.forecasts_df = forecasts_df
        self.actuals_df = actuals_df
        self.merged_df = None
        self.kpis_df = None

    @staticmethod
    def from_impala(impala_connect_cfg: dict, from_database: str, from_table: str) -> 'KPICalculator':
        """ Retrieves a scenario from Impala.

        :param impala_connect_cfg: Configuration to connect to Impala
        :param from_database: Database to read from
        :param from_table: Table in the database to read from
        :return: KPICalculator object containing the data from Impala
        """
        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_connect_cfg)

        # get forecasts
        req_forecasts = SQLUtils.get_fetch_latest(
            database_name=from_database,
            table_name=from_table,
            index_cols=[c for c in INDEX_COLS_FORECASTS if c != F_DN_FRC_MDF_DAT],
            version_col=F_DN_FRC_MDF_DAT,
            where_clause=f"WHERE `{F_DN_FRC_MTH_NBR}` > 0 AND `{F_DN_APO_FLG}`=TRUE"
        )

        logger.info(f'Reading {from_database}.{from_table}...')
        logger.debug('Request: %s' % SQLUtils.escape_req(req_forecasts))
        forecasts_df = pd.read_sql(sql=req_forecasts, con=impala_connection)
        forecasts_df[F_DN_MEA_DAT] = pd.to_datetime(forecasts_df[F_DN_MEA_DAT], format='%Y%m%d')

        # get actuals
        req_actuals = SQLUtils.get_fetch_latest(
            database_name=from_database,
            table_name=from_table,
            index_cols=INDEX_COLS_ACTUALS,
            version_col=F_DN_ETL_TST,
            where_clause=f"WHERE `{F_DN_FRC_FLG}`=FALSE"
        )
        logger.info(f'Reading {from_database}.{from_table}...')
        logger.debug('Request: %s' % SQLUtils.escape_req(req_actuals))
        actuals_df = pd.read_sql(sql=req_actuals, con=impala_connection)
        actuals_df[F_DN_MEA_DAT] = pd.to_datetime(actuals_df[F_DN_MEA_DAT], format='%Y%m%d')

        impala_connection.close()

        return KPICalculator(forecasts_df=forecasts_df, actuals_df=actuals_df)

    def to_impala(self, impala_connect_cfg, to_database: str, to_table: str) -> None:
        """ Saves KPIs to Impala
        :param impala_connect_cfg: Impala connection configuration
        :param to_database: Target database
        :param to_table: Target table
        :return:
        """
        if self.kpis_df is None:
            raise ValueError('No KPI table to export.')

        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_connect_cfg)
        impala_cursor = impala_connection.cursor()

        # we wipe the table before populating it again
        logger.info(f'Wiping `{to_database}`.`{to_table}`...')
        sql_delete_req = f"""
            TRUNCATE `{to_database}`.`{to_table}`
        """
        logger.debug(SQLUtils.escape_req(sql_delete_req))
        impala_cursor.execute(sql_delete_req)

        logger.info(f'Writing to `{to_database}`.`{to_table}`...')
        sql_insert_req = SQLUtils.make_insert_req_from_dataframe_infer_types(
            to_database=to_database,
            to_table=to_table,
            df=self.kpis_df
        )
        logger.debug(SQLUtils.escape_req(sql_insert_req[:1000]) + ' [...]')
        impala_cursor.execute(sql_insert_req)
        impala_connection.close()

    def calculate_kpis(self) -> None:
        self.merged_df = self.format_data()
        self.kpis_df = self.compute_kpis(self.merged_df, nb_month=12)

    def format_data(self):
        def group_dis_chl(df):
            return df.groupby(list(df.columns.difference(['dis_chl_cod', 'oft_trk_val', 'sal_ins_val'])), as_index=False).sum()

        # keeping relevant columns and dropping duplicates
        actuals_filtered = self.actuals_df.loc[:, self.rel_col].drop_duplicates()
        forecasts_filtered = self.forecasts_df.loc[:, self.rel_col].drop_duplicates()

        # computing maximum date for kpis
        date_max_kpi = actuals_filtered['mea_dat'].max()

        # filtering forecasts that will be used to compute KPIs
        forecasts_filtered = forecasts_filtered.loc[forecasts_filtered['mea_dat'] <= date_max_kpi]

        # grouping distribution channels
        actuals_filtered = group_dis_chl(actuals_filtered)
        forecasts_filtered = group_dis_chl(forecasts_filtered)

        # melting data
        id_vars_list = ['cyc_dat', 'lv2_umb_brd_cod', 'lv3_pdt_brd_cod', 'mat_cod', 'mea_dat']
        actuals_filtered = pd.melt(actuals_filtered, id_vars=id_vars_list)
        forecasts_filtered = pd.melt(forecasts_filtered, id_vars=id_vars_list)

        # renaming columns before merging
        actuals_filtered = actuals_filtered.rename(columns={'value': 'actual'})
        forecasts_filtered = forecasts_filtered.rename(columns={'value': 'forecast'})

        # dropping unmeaningful column for actuals
        actuals_filtered = actuals_filtered.drop(['cyc_dat'], axis=1)

        # merging actuals and forecasts
        grouping_columns = ['lv2_umb_brd_cod', 'lv3_pdt_brd_cod', 'mat_cod', 'mea_dat', 'variable']
        df_merged = actuals_filtered.merge(forecasts_filtered, on=grouping_columns, how='inner')

        # formating date
        # df_merged['mea_dat'] = df_merged['mea_dat'].apply(format_date)
        # df_merged['cyc_dat'] = df_merged['cyc_dat'].apply(format_date)

        # dropping useless lv2 column
        df_merged = df_merged.drop(['lv2_umb_brd_cod'], axis=1)

        # renaming columns
        df_merged = df_merged.rename(columns={'lv3_pdt_brd_cod': 'scope',
                                              'cyc_dat': 'cycle_month',
                                              'mat_cod': 'sku',
                                              'mea_dat': 'forecasted_month'})

        # dropping duplicates
        df_merged = df_merged.drop_duplicates()

        return df_merged

    def compute_kpis(self, df, nb_month):

        res = pd.DataFrame()
        for v in self.variable:
            temp = self.output_kpis(df, nb_month, variable=v)
            temp['variable'] = v
            res = pd.concat([res, temp])
        return res

    @staticmethod
    def output_kpis(df, nb_month, variable):

        def format_date(row):
            return row.strftime('%Y-%m')

        df_filtered = df[df['variable'] == variable].copy()

        horizons = range(1, nb_month)
        lv3_pdt_brd_cod = ['IL', 'DI', 'EIB', 'DC']
        list_date = df_filtered['forecasted_month'].unique()

        df_filtered['cycle_month'] = df_filtered['cycle_month'].apply(format_date)
        df_filtered['forecasted_month'] = df_filtered['forecasted_month'].apply(format_date)

        res = pd.DataFrame()
        for d in list_date:
            d = pd.to_datetime(d)
            for h in horizons:
                for c in lv3_pdt_brd_cod:
                    temp = pd.DataFrame(columns=['lv2_umb_brd_cod' , 'lv3_pdt_brd_cod', 'cyc_dat', 'kpi_type', 'prediction_horizon', 'value'])
                    month = format_date(d)
                    level = 'sku'
                    df_di_ = df_filtered[df_filtered['scope'] == c]
                    date_min = df_di_['cycle_month'].min()
                    date_min = pd.to_datetime(date_min, format='%Y-%m')

                    year_begining = d.year
                    first_month = max(
                        date_min + dateutil.relativedelta.relativedelta(months=h),
                        datetime.datetime(year_begining, 1, 1))

                    first_month_str = format_date(first_month)

                    try:
                        r_bias = kpis.YTD_rolling_bias(6, 3, month, df_di_, scope=c, agg_level='sku', verbose=True)
                    except:
                        r_bias = np.nan
                    try:
                        fa_ytd = kpis.YTD_FA(h, month, df_di_, scope=c, agg_level=level, first_month=first_month_str, verbose=True)
                        bias_ytd = kpis.YTD_bias(h, month, df_di_, scope=c, agg_level=level, first_month=first_month_str, verbose=True)
                    except:
                        fa_ytd = np.nan
                        bias_ytd = np.nan
                    try:
                        fa_monthly = kpis.YTD_FA(h, month, df_di_, scope=c, agg_level=level, first_month=month, verbose=True)
                        bias_monthly = kpis.YTD_bias(h, month, df_di_, scope=c, agg_level=level, first_month=month, verbose=True)
                    except:
                        fa_monthly = np.nan
                        bias_monthly = np.nan
                    temp['kpi_type'] = ['R6M', 'fa_ytd', 'bias_ytd', 'fa_monthly', 'bias_monthly']
                    temp['value'] = [r_bias, fa_ytd, bias_ytd, fa_monthly, bias_monthly]
                    temp['lv3_pdt_brd_cod'] = c
                    temp['cyc_dat'] = d
                    temp['prediction_horizon'] = h
                    if c == 'DC':
                        temp['lv2_umb_brd_cod'] = 'DC'
                    else:
                        temp['lv2_umb_brd_cod'] = 'IL'
                    res = pd.concat([res, temp])

        # get back to correct meaning of cyc_dat
        res['cyc_dat'] = pd.to_datetime(res['cyc_dat'], format='%Y-%m')
        res['cyc_dat'] = res.apply(
            lambda row: row['cyc_dat'] - dateutil.relativedelta.relativedelta(months=row['prediction_horizon']), axis=1)

        # fill nas
        res['value'].fillna(-99999, inplace=True)
        return res
