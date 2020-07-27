# coding: utf-8
import logging
import os,sys
import pandas as pd
import yaml
from datetime import datetime

from cfg.paths import DIR_CFG, DIR_CACHE, DIR_EXPORT_IMPALA_DATA, DIR_EXPORT_FEATURE_IMPORTANCE, DIR_EXPORT_FORECASTS, \
    RUN_DATETIME
from cfg.paths import LOG_FILENAME
from run import get_cycle_date
from src.data_wrangling import F_DI_TRADEFLOW, F_IL_OFFTAKE, F_IL_SELLIN, F_IL_OFFTAKE_CHANNEL, F_IL_MAP_SELLIN_EIB
from src.data_wrangling.Data import Data
from src.data_wrangling.RawMasterIL import RawMasterIL
from src.export_results.ExportFeatureImportance import ExportFeatureImportance
from src.forecaster.ForecasterIL import ForecasterIL
from src.postprocessing.postprocessing import PostProcessing
from src.scenario import F_RESULTS, F_DATABASE, F_TABLE, F_CONNECT, V_IL, F_FEATURE_IMPORTANCE
from src.scenario.Scenario import Scenario
from src.utils.logging import setup_loggers
from src.utils.misc import run_cmd

logger = logging.getLogger(__name__)
setup_loggers('demand_forecast_il', log_filepath=LOG_FILENAME)

DATA_FROM_CACHE = False
FORECASTS_FROM_CACHE = False
POSTPROC_FROM_CACHE = False
SCENARIO_FROM_CACHE = False
SAVE_TO_FS = True
SAVE_TO_IMPALA = True

# PATH_CSV_RESULTS = DIR_DATA  # todo Zhaoxia: set a path to store output as CSV


def run_il(date_start: int, horizon: int) -> None:
    """ Defines the flow to calculate IL forecasts based on the data available in Impala.

    :return:
    """
    date_start_dt = datetime(year=int(date_start / 100), month=date_start % 100, day=1)

    with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
        impala_cfg = yaml.load(ymlf)

    if DATA_FROM_CACHE:
        logger.info('Loading data from cache...')
        data_il = Data.from_pickle(os.path.join(DIR_CACHE, 'il_data.pkl'))
        raw_master = pd.read_csv(os.path.join(DIR_CACHE, 'raw_master_il.csv'), parse_dates=['date'])
        # raw_master = pd.read_pickle(os.path.join(DIR_CACHE, 'raw_master_il.pkl'))

    else:
        logger.info('Getting data from Impala...')
        data_il = Data.from_impala(label=V_IL, cfg=impala_cfg)
        data_il.to_pickle(os.path.join(DIR_CACHE, 'il_data.pkl'))
        raw_master = RawMasterIL(raw_data=data_il).df
        raw_master.to_csv(os.path.join(DIR_CACHE, 'raw_master_il.csv'), index=False)
        raw_master.to_pickle(os.path.join(DIR_CACHE, 'raw_master_il.pkl'))

    if FORECASTS_FROM_CACHE:
        logger.info('Loading forecasts from cache...')
        af_forecasts = pd.read_pickle(os.path.join(DIR_CACHE, 'af_forecast_il.pkl'))
        af_feature_importance = pd.read_pickle(os.path.join(DIR_CACHE, 'af_feature_importance_il.pkl'))

    else:
        logger.info('Running models...')
        forecaster = ForecasterIL.from_xgbparams(filename='forecaster_il.yaml')
        raw_master = raw_master.loc[raw_master['date'] <= date_start_dt, :]
        af_forecasts = forecaster.calculate_forecasts(date_start=date_start, horizon=horizon, raw_master=raw_master)
        af_feature_importance = forecaster.get_feature_importance()
        af_forecasts.to_pickle(os.path.join(DIR_CACHE, 'af_forecast_il.pkl'))
        af_forecasts.to_csv(os.path.join(DIR_CACHE, 'af_forecast_il.csv'))
        af_feature_importance.to_csv(os.path.join(DIR_CACHE, 'af_feature_importance_il.csv'))
        af_feature_importance.to_pickle(os.path.join(DIR_CACHE, 'af_feature_importance_il.pkl'))

    if POSTPROC_FROM_CACHE:
        logger.info('Loading postproc forecasts from cache')
        af_forecasts = pd.read_pickle(os.path.join(DIR_CACHE, 'il_fcst_split.pkl'))
        af_actuals = pd.read_pickle(os.path.join(DIR_CACHE, 'il_actual_split.pkl'))

    else:
        # post-processing
        logger.info('Post processing for granularity splits...')
        af_forecasts, af_actuals = PostProcessing.add_granularity_il(
            df_fcst_il_all=af_forecasts,
            df_hist_di=data_il[F_DI_TRADEFLOW],
            df_hist_il=data_il[F_IL_OFFTAKE_CHANNEL],
            df_hist_il_sellin=data_il[F_IL_SELLIN],
            df_hist_eib=data_il[F_IL_OFFTAKE],
            df_mapping=data_il[F_IL_MAP_SELLIN_EIB],
            ignore_hist_after_dt=date_start_dt
        )

        af_forecasts.to_csv(os.path.join(DIR_CACHE, 'il_fcst_split.csv'))
        af_actuals.to_csv(os.path.join(DIR_CACHE, 'il_actual_split.csv'))
        af_forecasts.to_pickle(os.path.join(DIR_CACHE, 'il_fcst_split.pkl'))
        af_actuals.to_pickle(os.path.join(DIR_CACHE, 'il_actual_split.pkl'))

    if SCENARIO_FROM_CACHE:
        logger.info('Loading scenario from cache')
        scenario = Scenario.from_pickle(os.path.join(DIR_CACHE, 'scenario_il.pkl'))

    else:
        scenario = Scenario.from_af_forecasts(af_forecasts=af_forecasts, af_actuals=af_actuals)
        scenario.to_pickle(os.path.join(DIR_CACHE, 'scenario_il.pkl'))

    if SAVE_TO_FS:
        data_il.to_csv(path=DIR_EXPORT_IMPALA_DATA)
        scenario.forecasts.to_csv(
            os.path.join(
                DIR_EXPORT_FORECASTS, f"{RUN_DATETIME.strftime('%Y%m%d%H%M%S')}_offtake-fcst_IL.csv"
            )
        )
        af_feature_importance.to_csv(
            os.path.join(
                DIR_EXPORT_FEATURE_IMPORTANCE, f"{RUN_DATETIME.strftime('%Y%m%d%H%M%S')}_feature-imptce_IL.csv"
            )
        )
        run_cmd(['chmod', '-R', 'ugo+rwx', DIR_EXPORT_IMPALA_DATA])
        run_cmd(['chmod', '-R', 'ugo+rwx', DIR_EXPORT_FORECASTS])
        run_cmd(['chmod', '-R', 'ugo+rwx', DIR_EXPORT_FEATURE_IMPORTANCE])

    if SAVE_TO_IMPALA:
        logger.info('Saving forecasts to Impala...')
        ExportFeatureImportance(
            feature_importance_df=af_feature_importance, cycle_date=date_start_dt, umb_label=V_IL
        ).to_impala(
            impala_connect_cfg=impala_cfg[F_CONNECT],
            to_database=impala_cfg[F_RESULTS][F_DATABASE],
            to_table=impala_cfg[F_FEATURE_IMPORTANCE][F_TABLE],
        )
        scenario.to_impala(
            impala_connect_cfg=impala_cfg[F_CONNECT],
            to_database=impala_cfg[F_RESULTS][F_DATABASE],
            to_table=impala_cfg[F_RESULTS][F_TABLE],
            comment='Base forecast'
        )
    logger.info('Done.')


if __name__ == '__main__':
    cyc_date = get_cycle_date(label=V_IL)
    run_il(date_start=int(cyc_date.strftime('%Y%m')), horizon=20)
    # run_il(date_start=201906, horizon=20)
