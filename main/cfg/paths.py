# encoding:utf-8
import os
from datetime import datetime

# root path of project (all programs must be run from root path)
DIR_ROOT = os.getcwd()

# paths to main folders
DIR_CFG = os.path.join(DIR_ROOT, 'cfg')
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
