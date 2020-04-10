# coding: utf-8
import logging
import os
import pickle
from datetime import datetime

import pandas as pd

from cfg.paths import DIR_CACHE
from src.scenario import *
from src.scenario import V_TINS, V_TONS
from src.utils.impala_utils import ImpalaUtils
from src.utils.sql_utils import SQLUtils

logger = logging.getLogger(__name__)


class Scenario:
    """ This object holds a scenario. This object is a key component in the back-office of the UI.
    """

    def __init__(self, user: str, creation_date: datetime, forecasts: pd.DataFrame, modification_date=None):
        self.user = user
        self.creation_date = creation_date
        self.modification_date = modification_date
        self.forecasts = forecasts

    @staticmethod
    def from_pickle(filename: str) -> 'Scenario':
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Loading {__class__} from pickle {fpath}...')
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, filename: str) -> None:
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Saving {__class__} to pickle {fpath}...')
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def from_af_forecasts(
            af_forecasts: pd.DataFrame, af_actuals: pd.DataFrame
    ) -> 'Scenario':
        """ Creates a scenario from AF results.

        :param af_forecasts: Output of the AF model for IL
        :param af_actuals: Master data table containing the input data (Interface contracts)
        :return: Scenario containing the AF results
        """
        actuals_af = af_actuals.copy(deep=True)
        actuals_af.rename(columns=MAPPING_AF_TO_DN, inplace=True)
        actuals_af[F_DN_FRC_FLG] = False

        forecasts_af = af_forecasts.copy(deep=True)
        forecasts_af.rename(columns=MAPPING_AF_TO_DN, inplace=True)
        forecasts_af[F_DN_FRC_FLG] = True

        scenario_forecasts: pd.DataFrame = pd.concat([forecasts_af, actuals_af], sort=False)
        scenario_forecasts = Scenario.remove_overlap(scenario_forecasts=scenario_forecasts)

        scenario_forecasts[F_DN_OFT_VAL] = scenario_forecasts[F_DN_OFT_TRK_VAL]
        scenario_forecasts[F_DN_SAL_OUT_VAL] = 0
        scenario_forecasts[F_DN_SAL_INS_VAL] = 0
        scenario_forecasts[F_DN_RTL_IVT_VAL] = 0
        scenario_forecasts[F_DN_SUP_IVT_VAL] = 0
        scenario_forecasts[F_DN_RTL_IVT_COV_VAL] = 0
        scenario_forecasts[F_DN_SUP_IVT_COV_VAL] = 0

        scenario_forecasts.sort_values(by=F_DN_MEA_DAT, inplace=True)
        scenario_forecasts.fillna(value=0.0, inplace=True)
        scenario_forecasts.reset_index(drop=True, inplace=True)

        Scenario.detect_dups(df=scenario_forecasts)

        return Scenario(user=V_AF, creation_date=datetime.today(), forecasts=scenario_forecasts)

    @staticmethod
    def from_impala(user: str, creation_date: datetime, modification_date: datetime,
                    impala_connect_cfg: dict, from_database: str, from_table: str) -> 'Scenario':
        """ Retrieves a scenario from Impala.

        :param user: User name associated to the scenario
        :param creation_date: Creation date of the scenario
        :param modification_date: Modification date to select
        :param impala_connect_cfg: Configuration to connect to Impala
        :param from_database: Database to read the scenario from
        :param from_table: Table in the database to read the scenario from
        :return: Scenario containing the results selected from Impala
        """
        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_connect_cfg)

        req = f"""
            SELECT * FROM `{from_database}`.`{from_table}`
            WHERE `{F_DN_FRC_USR_NAM_DSC}`='{user}'
             AND `{F_DN_FRC_CRE_DAT}`='{creation_date.strftime(F_DN_DATE_FMT)}'
             AND `{F_DN_FRC_MDF_DAT}`='{modification_date.strftime(F_DN_DATE_FMT)}'
        """

        logger.info(f'Reading {from_database}.{from_table}...')
        logger.debug('Request: %s' % req.replace("\n", " "))

        forecasts = pd.read_sql(sql=req, con=impala_connection)
        forecasts[F_DN_MEA_DAT] = pd.to_datetime(forecasts[F_DN_MEA_DAT], format='%Y%m%d')

        impala_connection.close()

        return Scenario(user=user, creation_date=creation_date, forecasts=forecasts,
                        modification_date=modification_date)

    def to_impala(self, impala_connect_cfg, to_database: str, to_table: str, comment: str = '') -> None:
        """ Stores scenario to Impala

        :param comment: Comment associated to scenario
        :param impala_connect_cfg: Configuration to connect to Impala
        :param to_database: Database to write the scenario to
        :param to_table: Table in the database to write the scenario to
        :return: Nothing. Data written in Impala
        """
        impala_connection = ImpalaUtils.get_impala_connector(cfg=impala_connect_cfg)
        impala_cursor = impala_connection.cursor()

        forecasts: pd.DataFrame = self.forecasts.copy(deep=True)

        now = datetime.today()
        self.modification_date = now
        cyc_dat = forecasts.loc[forecasts[F_DN_FRC_MTH_NBR] == 0, F_DN_MEA_DAT].iloc[0]
        forecasts[F_DN_CYC_DAT] = cyc_dat

        if forecasts[F_DN_LV2_UMB_BRD_COD].iloc[0] == V_DC:
            forecasts[F_DN_UNIT] = V_TINS
        elif forecasts[F_DN_LV2_UMB_BRD_COD].iloc[0] == V_IL:
            forecasts[F_DN_UNIT] = V_TONS
        else:
            raise AssertionError(f"Value {forecasts[F_DN_LV2_UMB_BRD_COD].iloc[0]} for {F_DN_LV2_UMB_BRD_COD} unknown.")

        forecasts[F_DN_FRC_USR_NAM_DSC] = self.user
        forecasts[F_DN_FRC_CRE_DAT] = self.creation_date.strftime(F_DN_DATE_FMT)
        forecasts[F_DN_FRC_MDF_DAT] = now.strftime(F_DN_DATE_FMT)
        forecasts[F_DN_LV6_PDT_NAT_COD] = forecasts[F_DN_LV6_PDT_NAT_COD].apply(str)
        forecasts[F_DN_APO_FLG] = False
        forecasts[F_DN_USR_NTE_TXT] = comment
        forecasts[F_DN_MEA_DAT] = pd.to_numeric(
            forecasts[F_DN_MEA_DAT].apply(lambda x: x.strftime('%Y%m%d')), downcast='integer'
        )
        forecasts[F_DN_FRC_MTH_NBR] = pd.to_numeric(forecasts[F_DN_FRC_MTH_NBR], downcast='integer')

        forecasts[F_DN_ETL_TST] = now.strftime(F_DN_DATE_FMT)

        forecasts.fillna(value=0, inplace=True)
        forecasts.reset_index(drop=True, inplace=True)

        logger.info(f'Writing to `{to_database}`.`{to_table}`...')
        chunk_size = 10000
        for i in range(0, len(forecasts.index), chunk_size):
            logger.info(f'Inserting chunk {i}')
            chunk_df = forecasts.iloc[i:i + chunk_size]
            sql_insert_req = SQLUtils.make_insert_req_from_dataframe_infer_types(
                to_database=to_database, to_table=to_table, df=chunk_df
            )
            logger.debug(SQLUtils.escape_req(sql_insert_req[:1000]) + ' [...]')
            impala_cursor.execute(sql_insert_req)

        impala_connection.close()

    @staticmethod
    def detect_dups(df: pd.DataFrame) -> None:

        check_dups = df.groupby(V_DN_LOWEST_GRANULARITY + [F_DN_MEA_DAT]).agg({
            F_DN_OFT_VAL: 'count'
        }).reset_index(inplace=False)

        dups = check_dups.loc[check_dups[F_DN_OFT_VAL] > 1]
        if len(dups.index) > 0:
            raise ValueError(f'Forecast data contains duplicate.\n{dups.transpose()}')

    @staticmethod
    def remove_overlap(scenario_forecasts: pd.DataFrame) -> pd.DataFrame:

        earliest_forecast_date = scenario_forecasts.loc[scenario_forecasts[F_DN_FRC_FLG], F_DN_MEA_DAT].min()
        scenario_forecasts = pd.concat([
            scenario_forecasts.loc[
                (scenario_forecasts[F_DN_MEA_DAT] >= earliest_forecast_date) & (scenario_forecasts[F_DN_FRC_FLG])
                ].copy(),
            scenario_forecasts.loc[
                (scenario_forecasts[F_DN_MEA_DAT] < earliest_forecast_date) & (~scenario_forecasts[F_DN_FRC_FLG])
                ].copy(),
        ])

        return scenario_forecasts
