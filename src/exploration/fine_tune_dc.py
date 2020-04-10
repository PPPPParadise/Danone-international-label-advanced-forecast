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
from src.forecaster.modeldc import Modeldc, ModelConfig

warnings.simplefilter(action='ignore', category=FutureWarning)

uri = "mongodb://%s:%s@%s" % (
    quote_plus("root"),
    quote_plus("example"),
    "localhost:27017"
)
mongo_client = MongoClient(uri)
db = mongo_client["fine_tune_dc"]
collection = db["config_errors"]


def run_model(features, model_config):
    print(f"..runnning with config : {model_config}")
    table_all_features = features.copy()

    dwp_test = util.create_list_period(201710, 201902, False)

    res = mod.recreate_past_forecasts(table_all_features, dwp_test, model_config, horizon=8)
    test = diagnostic.Diagnostic(cvr=res[res.horizon == 6], raw_master=raw_master,
                                 postprocess='indep', di_eib_il_format=False)
    temp = test.run_test_dc(plot=False, horizon=6)

    error = (
        temp
        .assign(abs_error=lambda x: abs(x["prediction"] - x["target_dc"]))
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
    raw_master = pd.read_csv('data/raw/raw_master_dc_20191016.csv')
    mod = Modeldc(raw_master)
    max_date_available = mod.all_sales.calendar_yearmonth.max()
    filter_date = min(201908, max_date_available)
    dwps = util.create_list_period(201701, filter_date, False)
    dwp, dtp = util.get_all_combination_date(dwps, 10)
    features = mod.create_all_features(dwp, dtp)

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
            to_save = deepcopy(model_config)
            to_save["error"] = error

            collection.insert_one(to_save)
        except Exception as error:
            print("Error on config")
            print(model_config)
            print(str(error))

            to_save = deepcopy(model_config)
            to_save["error"] = -1
            to_save["error_message"] = str(error)
            collection.insert_one(to_save)

            continue
