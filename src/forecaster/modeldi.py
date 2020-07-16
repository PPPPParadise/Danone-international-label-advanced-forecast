from functools import reduce
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor
from cfg.paths import DIR_TEST_DATA
from src.forecaster.MLDCModel import MLDCModel
from src.forecaster.features import *
from src.forecaster.model import Model
from src.forecaster.utilitaires import *
from collections import namedtuple

DEFAULT_DATA_LAG_IL = 2

ModelConfig = namedtuple("ModelConfig", "model_name model_params")

class Modeldi(Model):
    """
    XGB Regressor model with postprocessing of the output
    One XGB model is trained for each prediction horizon
    """

    def __init__(self, package_data_di, sell_in_fc, sell_in_actual, eln_fc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = False
        self.package_data_di = package_data_di
        self.sell_in_fc = sell_in_fc
        self.sell_in_actual = sell_in_actual
        self.eln_fc = eln_fc
        if self.debug:
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")
            print("       DEBUG       ")
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")
        self.format_additional_tables()

    def preformat_table(self):
        """ Preformat data table to feed into model
        This function allows to precompute useful tables that will be used by the model
        """

        sales_danone = self.sales_danone.copy()

        # 1. Formatting DI data
        sales_danone_di = format_label_data(sales_danone, 'di')

        # 2. Formatting EIB data
        sales_danone_eib = format_label_data(sales_danone, 'eib')

        # 3. Formatting IL data
        sales_danone_il = format_label_data(sales_danone, 'il')

        # 4. Merging data
        all_sales = pd.concat([sales_danone_eib, sales_danone_di, sales_danone_il])

        # 5. Selecting relevant columns
        all_sales = sales_danone_di[
            ['calendar_yearmonth', 'sku_wo_pkg', 'country', 'brand', 'tier',
             'stage', 'label', 'offtake', 'sellin', 'sellout']]

        self.all_sales = all_sales
        all_sales.to_pickle(os.path.join(DIR_TEST_DATA, 'test_all_sales_il.pkl'))

    def format_additional_tables(self):
        """ Preformat data table to feed into model
        This function allows to precompute useful tables that will be used by the model
        """
        # package sales

        self.package_data_di['sku_w_pkg'] = self.package_data_di.sku_wo_pkg.str.split('_').str[0:3].str.join('_') + '_'\
                                            + self.package_data_di.sku_with_pkg + '_'+ \
                                            self.package_data_di.sku_wo_pkg.str.split('_').str[3]

        self.package_data_di['calendar_yearmonth'] = pd.to_datetime(self.package_data_di['date']).dt.year.astype(
            str) + pd.to_datetime(self.package_data_di['date']).dt.month.astype(str).str.zfill(2)
        self.package_data_di['calendar_yearmonth'] = self.package_data_di['calendar_yearmonth'].astype(int)
        self.package_data_di['label'] = self.package_data_di['label'].str.lower()

        # sell in actuals
        self.sell_in_actual['month'] = pd.to_datetime(self.sell_in_actual['month'], format="%Y-%m-%d")
        self.sell_in_actual['calendar_yearmonth'] = pd.to_datetime(self.sell_in_actual['month']).dt.year.astype(
            str) + pd.to_datetime(self.sell_in_actual['month']).dt.month.astype(str).str.zfill(2)
        self.sell_in_actual['calendar_yearmonth'] = self.sell_in_actual['calendar_yearmonth'].astype(int)
        self.sell_in_actual.rename(columns={'SKU_AF': 'sku_w_pkg'}, inplace=True)

        # sell in forecast
        self.sell_in_fc['cycle_month'] = pd.to_datetime(self.sell_in_fc['cycle_month'], format="%Y-%m")
        self.sell_in_fc['cycle_month'] = self.sell_in_fc.cycle_month - pd.DateOffset(months=3)
        self.sell_in_fc['cycle_month'] = pd.to_datetime(self.sell_in_fc['cycle_month']).dt.year.astype(
            str) + pd.to_datetime(self.sell_in_fc['cycle_month']).dt.month.astype(str).str.zfill(2)
        self.sell_in_fc['cycle_month'] = self.sell_in_fc['cycle_month'].astype(int)
        self.sell_in_fc['calendar_yearmonth'] = self.sell_in_fc.forecasted_month
        self.sell_in_fc['calendar_yearmonth'] = pd.to_datetime(self.sell_in_fc['calendar_yearmonth']).dt.year.astype(
            str) + pd.to_datetime(self.sell_in_fc['calendar_yearmonth']).dt.month.astype(str).str.zfill(2)
        self.sell_in_fc['calendar_yearmonth'] = self.sell_in_fc['calendar_yearmonth'].astype(int)

        # eln forecast
        self.eln_fc['cycle_month'] = pd.to_datetime(self.eln_fc['cycle_month'], format="%Y-%m")
        self.eln_fc['cycle_month'] = self.eln_fc.cycle_month - pd.DateOffset(months=2)
        self.eln_fc['cycle_month'] = pd.to_datetime(self.eln_fc['cycle_month']).dt.year.astype(
            str) + pd.to_datetime(self.eln_fc['cycle_month']).dt.month.astype(str).str.zfill(2)
        self.eln_fc['cycle_month'] = self.eln_fc['cycle_month'].astype(int)
        self.eln_fc['calendar_yearmonth'] = self.eln_fc.forecasted_month
        self.eln_fc['calendar_yearmonth'] = pd.to_datetime(self.eln_fc['calendar_yearmonth']).dt.year.astype(
            str) + pd.to_datetime(self.eln_fc['calendar_yearmonth']).dt.month.astype(str).str.zfill(2)
        self.eln_fc['calendar_yearmonth'] = self.eln_fc['calendar_yearmonth'].astype(int)

    def create_all_features(self, dwp, dtp):
        """ Creates all features
        """

        # SALES wo pckg

        # 1.a Creates sales feature for country-brand-tier-stage-label granularity
        grouped_offtake2 = \
            self.all_sales.groupby(['calendar_yearmonth', 'country', 'brand', 'tier', 'stage', 'label', 'sku_wo_pkg'])[
                'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_stage_label_', timelag=3)

        # 2. Creates sales feature for country-brand-tier-stage granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df2 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_stage_', timelag=3)

        # SELL OUT wo pckg

        # 6. Creates sellout feature for sku-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg', 'label'])[
            'sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df3 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellout_sku_label_', timelag=3)

        # 16. Creates seasonality feature
        df16 = create_seasonality_features(dtp)

        # Sales w pckg

        grouped_offtake2 = self.package_data_di.groupby(['calendar_yearmonth', 'country', 'brand', 'tier', 'stage',
                                                         'label', 'sku_wo_pkg', 'sku_w_pkg'])['offtake']\
            .sum().unstack('calendar_yearmonth').fillna(0)
        df4 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sales_sku_w_pkg_', timelag=4)

        # 1.b Creates target
        target = features_target(grouped_offtake2, dwp, dtp)

        # Sellin act w pkg
        grouped_offtake2 = self.sell_in_actual.groupby(['calendar_yearmonth', 'sku_w_pkg'])[
            'sum(a.actual)'].sum().unstack('calendar_yearmonth').fillna(0)
        df5 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_sku_w_pkg_', timelag=3)

        # Sell in fc
        df6 = features_sell_in_fc(self.sell_in_fc, ['sku_w_pkg', 'calendar_yearmonth'], dwp, dtp, delta_window=5)

        # Sell in fc
        df7 = features_eln_fc(self.eln_fc, ['sku_w_pkg', 'calendar_yearmonth'], dwp, dtp, delta_window=3)

        # the first dataframe should have all granularities
        # dfs = [df4, df2, df3, df1, df5, df6, df7, df16, target]
        dfs = [df4, df2, df1, df6, df7, target]
        # dfs = [df4, df7, target]

        # Merging features
        dffinal = reduce(lambda left, right: pd.merge(left, right, on=list(
            {'date_when_predicting', 'date_to_predict', 'country', 'brand', 'tier', 'stage', 'label',
             'sku_wo_pkg', 'sku_w_pkg'} & set(right.columns)), how='left'), dfs)

        # Adding onehot encoding labels
        one_hot = pd.get_dummies(dffinal['label'])
        dffinal = pd.concat([dffinal, one_hot], axis=1)

        return dffinal, dfs

    def correct_fc(self, res, month_to_correct=('CNY', 11), thrsh=0.05):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return:
        """

        sales = self.all_sales.groupby(['calendar_yearmonth', 'label'])['offtake'].sum().reset_index()
        temp = res.copy()
        years = list((res.date_to_predict // 100).unique())
        tempdi = temp[temp.label == 'di']

        for y in years:
            for m in month_to_correct:
                # Post-process for di
                tempdc = apply_forecast_correction(sales=sales, forecast=temp, forecast_filtered=tempdi, label='di',
                                                   year=y, month=m, thrsh=thrsh)

        return tempdi

    def forecast_since_date_at_horizon(self, date_start, horizon, params):
        """ Function that performs a full forecast since a date of sales
        :param date_start: last date of available sales in the data that need to be used by the model when forecasting
        :param horizon: horizon in the future for which we want a forecast
        :param params: parameters of the xgboost model
        :return: a dataframe containing the forecast
        """

        max_date_available = self.all_sales.calendar_yearmonth.max()
        filter_date = min(date_start, max_date_available)
        dwps = create_list_period(201601, filter_date, False)
        dwp, dtp = get_all_combination_date(dwps, horizon)

        print("creating the main table")
        table_all_features = self.create_all_features(dwp, dtp)
        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        features_int = ["date_when_predicting", "label", "date_to_predict", "sku_wo_pkg", "target", "country", "brand",
                        "tier", "stage", "horizon"]
        features = [x for x in table_all_features.keys() if x not in features_int]
        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()

        for h in range(1, horizon + 1):
            print("training model at horizon: " + str(h - DEFAULT_DATA_LAG_IL))
            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())]
            x_train = subdata[features].values
            y_train = subdata.target
            data_test = table_all_features[(table_all_features.date_when_predicting == filter_date) &
                                           (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values
            model = xgb.XGBRegressor(**params)

            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            preds = preds.clip(min=0)

            data_test['prediction'] = preds
            #data_test['prediction'] = data_test['forecast']
            data_test['horizon'] = h
            res = pd.concat([res, data_test[["label", "date_to_predict", "sku_wo_pkg", "horizon", "prediction"]]])

            # Creating feature importance
            feature_importance = dict(zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(), ignore_index=True)

        self.feature_importance = feature_importance_df

        res.to_pickle(os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_il.pkl'))

        # Rescaling
        res = self.rescale_il_dieib(res)

        # Applying post-processing
        res = self.correct_fc(res, month_to_correct=['CNY', 11])

        return res

    def recreate_past_forecasts(self, table_all_features, list_dwps, horizon=10, model_config=None):

        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        features_int = ["date_when_predicting", "label", "date_to_predict", "sku_wo_pkg", "target", "country", "brand",
                        "tier", "stage", "horizon", "sku_w_pkg"]
        features = [x for x in table_all_features.keys() if x not in features_int]

        table_all_features = table_all_features[~table_all_features.forecast_eln.isnull()]

        # 3. Prediction at sku granularity di and ieb
        # print("Training SKU model for IEB and DI")
        resfinal = pd.DataFrame()
        for datwep in list_dwps:
            print(f".. date when predicting {datwep}")
            res = pd.DataFrame()
            for h in range(1, horizon + 1):
            #for h in range(5, 8):
                print(f".... horizon {h}")
                subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                             & (table_all_features.date_to_predict <= datwep)]
                if subdata.shape[0] < 1:
                    continue
                x_train = subdata[features].values
                y_train = subdata.target

                data_test_di = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                                   (table_all_features.horizon == h) &
                                                   (table_all_features.label == 'di')]

                x_test_di = data_test_di[features].values

                if not self.debug:

                    # PARAMS = params_dict_di[h]
                    # Instantiate model
                    # model = xgb.XGBRegressor(**PARAMS)
                    # model = ExtraTreesRegressor(**PARAMS)
                    # where_are_NaNs = np.isnan(x_train)
                    # x_train[where_are_NaNs] = 0
                    model = MLDCModel(
                        model_name=model_config.model_name,
                        model_params=model_config.model_params
                    )
                    model.fit(x_train, y_train)

                    # where_are_NaNs = np.isnan(x_test_di)
                    # x_test_di[where_are_NaNs] = 0
                    preds_di = model.predict(x_test_di)

                else:
                    # dummy prediction
                    preds_di = np.ones(len(x_test_di)) * 3000

                preds_di = preds_di.clip(0)

                data_test_di = data_test_di.assign(
                    horizon=h,
                    prediction=preds_di
                )
                data_test = data_test_di
                # data_test['prediction'] = data_test['forecast']
                res = pd.concat([res, data_test[["label", "date_to_predict", "sku_wo_pkg", "sku_w_pkg", "prediction", "date_when_predicting", "horizon"]]])

            # rescale il di eib
            #res = self.correct_fc(res, month_to_correct=['CNY', 11], thrsh=0.05)

            resfinal = pd.concat([resfinal, res])

        return resfinal


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import src.forecaster.utilitaires as util

    raw_master = pd.read_csv('data/raw/raw_master_il_0910.csv')
    mod = Modeldi(raw_master)
    dwp_test = util.create_list_period(201707, 201902, False)

    max_date_available = mod.all_sales.calendar_yearmonth.max()
    filter_date = min(201908, max_date_available)
    dwps = util.create_list_period(201601, filter_date, False)
    dwp, dtp = util.get_all_combination_date(dwps, 10)

    # 1. Read precalculated features
    table_all_features = pd.read_csv('data/table_all_features.csv')

    # 2. Remove negative targets
    msk = table_all_features.target >= 0
    print(
        f"removing {len(msk) - sum(msk)}/{len(msk)} rows because of a negative target")
    table_all_features = table_all_features[msk]

    XGBOOST_PARAMETERS = {
        'max_depth': 11,  # 25
        'gamma': 0.02,
        'subsample': 0.4,
        'n_estimators': 27,
        'learning_rate': 0.1,
        'n_jobs': 12,
        'verbosity': 2
    }
    dico_params = {}
    for i in range(1, 11):
        dico_params[i] = XGBOOST_PARAMETERS.copy()

    dico_params[5]['n_estimators'] = 27
    dico_params[6]['n_estimators'] = 31
    dico_params[7]['n_estimators'] = 31
    dico_params[8]['n_estimators'] = 47
    dico_params[9]['n_estimators'] = 13
    dico_params[10]['n_estimators'] = 9


    # 3. run model
    res = mod.recreate_past_forecasts(table_all_features, dwp_test, dico_params)
