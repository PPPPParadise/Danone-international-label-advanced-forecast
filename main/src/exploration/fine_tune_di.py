import yaml
import random
import pandas as pd
from copy import deepcopy
from itertools import product
from collections import OrderedDict
from pymongo import MongoClient
from urllib.parse import quote_plus
import warnings

import src.forecaster.utilitaires as util
import src.forecaster.diagnostic as diagnostic
from src.forecaster.modeldi import Modeldi, ModelConfig

warnings.simplefilter(action='ignore', category=FutureWarning)

uri = "mongodb://%s:%s@%s" % (
    quote_plus("root"),
    quote_plus("example"),
    "localhost:27017"
)
mongo_client = MongoClient(uri)
db = mongo_client["fine_tune_di"]
collection = db["fa_errors"]


def run_model(features, model_config):
    print(f"..runnning with config : {model_config}")
    table_all_features = features.copy()

    dwp_test = util.create_list_period(201702, 201902, False)

    res = mod.recreate_past_forecasts(table_all_features, dwp_test, model_config=model_config, horizon=8)

    res2 = res.groupby(['date_to_predict', 'sku_wo_pkg', 'horizon'])['prediction'].sum().reset_index()

    res_di = res2.copy()
    res_eib = res2.copy()
    res_il = res2.copy()
    res_di['label'] = 'di'
    res_eib['label'] = 'eib'
    res_il['label'] = 'il'

    res_final = pd.concat([res_il, res_eib, res_di])
    res_final['ratio'] = 1

    test = diagnostic.Diagnostic(cvr=res_final[(res_final.horizon == 6)], raw_master=raw_master, postprocess='indep')
    temp = test.run_test(plot=False, prediction_horizon=6)

    error = (
        temp
        .assign(abs_error=lambda x: abs(x["yhat_di_calib"] - x["target_di"]))
        .abs_error
        .sum()
    )

    return error


def prepare_configs(meta_config):
    """
    Returns list of model_configs
    :param meta_config:
    :return:
    """

    configs = []

    for model_name, list_config in meta_config.items():

        combinations = product(*list(list_config.values()))

        combinations = [
            {
                parameter_name: parameter_value
                for parameter_name, parameter_value
                in zip(list_config.keys(), combination)
            }
            for combination in combinations
        ]

        for model_params in combinations:
            configs.append(ModelConfig(
                model_name=model_name,
                model_params=deepcopy(model_params)
            ))

    return configs


if __name__ == '__main__':

    # 0. create table all features
    sell_in_fc = pd.read_csv('data/raw/DI/DI_sellin_forecast_full.csv')
    raw_master = pd.read_csv('data/raw/raw_master_il_1019.csv')
    package_data = pd.read_csv('data/raw/DI/il_actual_split_201909.csv')
    package_data_di = package_data.query("label == 'DI'").copy()
    sell_in_actual = pd.read_csv('data/raw/DI/di sellin actual.csv')
    fc_eln = pd.read_csv('data/raw/DI/di_forecast.csv')

    mod = Modeldi(package_data_di, sell_in_fc, sell_in_actual, fc_eln, raw_master)
    max_date_available = mod.all_sales.calendar_yearmonth.max()
    filter_date = min(201908, max_date_available)
    dwps = util.create_list_period(201605, filter_date, False)
    dwp, dtp = util.get_all_combination_date(dwps, 10)

    features, all_df = mod.create_all_features(dwp, dtp)
    features.fillna(0, inplace=True)

    # 1. read meta prams config
    with open("src/exploration/models.yaml", "r") as stream:
        meta_config = OrderedDict(yaml.load(stream=stream))

    # 2. prepare configs
    model_configs = prepare_configs(meta_config=meta_config)
    random.shuffle(model_configs)
    print(f"Running model on {len(model_configs)} models")

    # 3. run the model for each config
    for (ix, model_config) in enumerate(model_configs):
        print(f"Running model {ix+1}/{len(model_configs)}")
        try:
            # run model
            error = run_model(features, model_config)

            # save result - config and error
            #to_save = deepcopy(model_config)
            to_save = {
                "model_name": model_config.model_name,
                "model_params": deepcopy(model_config.model_params)
            }

            to_save["error"] = error

            collection.insert_one(to_save)
        except Exception as error:
            print("Error on config")
            print(model_config)
            print(str(error))


            to_save = {
                "model_name": model_config.model_name,
                "model_params": deepcopy(model_config.model_params)
            }

            to_save["error"] = -1
            to_save["error_message"] = str(error)
            collection.insert_one(to_save)

            continue
