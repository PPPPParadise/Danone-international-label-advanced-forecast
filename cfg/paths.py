# encoding:utf-8
import os
from datetime import datetime
from collections import namedtuple


# root path of project (all programs must be run from root path)
DIR_ROOT = os.getcwd()

# paths to main folders
DIR_CFG = os.path.join(DIR_ROOT, 'cfg')
DIR_TEM = os.path.join(DIR_ROOT, 'temp')
DIR_DATA = os.path.join(DIR_ROOT, 'data')
DIR_LOGS = os.path.join(DIR_ROOT, 'logs')
DIR_CACHE = os.path.join(DIR_ROOT, '.cache')
DIR_TEST = os.path.join(DIR_ROOT, 'tests')
DIR_TEST_DATA = os.path.join(DIR_TEST, 'test_data')
DIR_TEST_OUTPUT = os.path.join(DIR_TEST, 'test_output')

DIR_MAPPINGS = os.path.join(DIR_CFG, 'mappings')
FILEPATH_MAPPING_APO_CHANNEL_NFA_RFA = os.path.join(DIR_MAPPINGS, 'CN3_AF_APO_channel_nfa_rfa.csv')
FILEPATH_MAPPING_APO_STAGE_SKU_SPLIT_DC = os.path.join(DIR_MAPPINGS, 'CN3_AF_APO_stage_sku_split_7851.csv')
FILEPATH_MAPPING_APO_STAGE_SKU_SPLIT_DI = os.path.join(DIR_MAPPINGS, 'CN3_AF_APO_stage_sku_split_7871.csv')
FILEPATH_MAPPING_APO_SCOPE = os.path.join(DIR_MAPPINGS, 'CN3_AF_APO_labels.csv')
FILEPATH_MAPPING_DI_SKU = os.path.join(DIR_MAPPINGS, 'M_DI.csv')

RUN_DATETIME = datetime.today().strftime("%Y-%m-%d_%H%M%S")
LOG_FILENAME = os.path.join(DIR_LOGS, f'{RUN_DATETIME}_%s.log')
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'

# paths to data_prep config files
CFG_DATA_PREP = os.path.join(DIR_CFG, 'data_prep.yml')
CFG_ADDL_GRAN = os.path.join(DIR_CFG, 'addl_gran.yml')

# APO export path
FILEPATH_APO_TARGET = '/pctmp/smartdata/interf/APO_sim'
# FILEPATH_APO_TARGET = '/pctmp/smartdata/interf/APO_prod'

# Cache
FILENAME_LRA = 'lra.pkl'
FILENAME_RAW_MASTER_IL = 'raw_master_il.csv'
FILENAME_RAW_MASTER_DC = 'raw_master_dc.csv'

# Locks
FILENAME_LOCK_SCENARIOS = 'lock_scenarios.pkl'

# Add by Artefact
config = {}

## file path
config["data_folder_path"] = "../data"
config["temp_folder_path"] = "../temp"
config["result_folder_path"] = "../results"

## input files
config["input_raw_master"] = "raw_master_il_20200210_Update_0602_subbrand_catupdate_v2+2020.csv"
config["input_category_forecast"] = "IL_feature_table_all_0610_cat_fsct.csv"

## temp files
config["feature_import_first_run"] = "feature_importance_df_sets_0610.csv"

## Result file
config['IL_forecast_result'] = "IL_forecast_.csv"

config["FirstRun"] = False 
config['horizon'] = 12

## Columns configuration
config["features_int"] = ["date_when_predicting", 
                          "label",
                          "date_to_predict",
                          "target",
                          "country", 
                          "brand",
                          "horizon",
                          "country_brand_channel",
                          "country_brand"]

config["features_pop_col"] = ['0to6_month_population_mean_3M',
                              '6to12_month_population_mean_3M',
                              '12to36_month_population_mean_3M',
                              '0to6_month_population_mean_6M',
                              '6to12_month_population_mean_6M',
                              '12to36_month_population_mean_6M',
                              '0to6_month_population_mean_9M',
                              '6to12_month_population_mean_9M',
                              '12to36_month_population_mean_9M',
                              '0to6_month_population_mean_12M',
                              '6to12_month_population_mean_12M',
                              '12to36_month_population_mean_12M']   

config["features_cat_fsct_col"] = ['upre_fsct',
                                   'spre_fsct',
                                   'mainstream_fsct',
                                   'upre_mean_3M_fsct',
                                   'spre_mean_3M_fsct',
                                   'mainstream_mean_3M_fsct'] 

## Model Parameters
ModelConfig = namedtuple("ModelConfig", "model_name model_params")

config["model_config_XGBRegressor"] =  ModelConfig(
    model_name="XGBRegressor",
    model_params={
        'max_depth': 8,
        'gamma': 0.02,
        'subsample': 0.3,
        'n_estimators': 60,
        'learning_rate': 0.1,
        'n_jobs': 12,
        'verbosity': 2})

config["model_config_RandomForestRegressor"] = ModelConfig(
        model_name="RandomForestRegressor",
        model_params={
            'max_depth': 8,
            'n_estimators': 80,
            'max_features':50,
            'n_jobs': -1}) 

config["model_config_ExtraTreesRegressor"] = ModelConfig(
        model_name="ExtraTreesRegressor",
        model_params={
            'max_depth': 8,
            'n_estimators': 60,
            'max_features':50,
            'n_jobs': 12}) 

config["model_config_AdaBoostRegressor"] = ModelConfig(
        model_name="AdaBoostRegressor",
        model_params={
            'n_estimators': 80,
            'learning_rate': 0.2,
            'loss':'square'}) 

config["model_config_GradientBoostingRegressor"] = ModelConfig(
        model_name="GradientBoostingRegressor",
        model_params={ 
            'subsample': 0.3,
            'n_estimators': 80,
            'learning_rate': 0.1})