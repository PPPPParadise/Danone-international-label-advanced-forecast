from functools import reduce
from copy import deepcopy
from collections import namedtuple
from cfg.paths import DIR_TEST_DATA
from src.forecaster.MLDCModel import MLDCModel
from src.forecaster.features import *
from src.forecaster.model import Model
from src.forecaster.utilitaires import *

ModelConfig = namedtuple("ModelConfig", "model_name model_params")


class Modeldc(Model):
    """
    XGB Regressor model with postprocessing of the output
    One XGB model is trained for each prediction horizon
    """

    def __init__(self, data):
        super(Modeldc, self).__init__(data=data)

        self.default_model_config = ModelConfig(
            model_name="GradientBoostingRegressor",
            model_params={
                'standard_scaling': False,
                'pca': 0,
                'loss': 'huber',
                'learning_rate': 0.01,
                'n_estimators': 500,
                'subsample': 0.3,
                'max_depth': 10,
                'max_features': 0.7,
                'alpha': 0.9,
                'random_state': 42
            }
        )

    def preformat_table(self):
        """ Preformat data table to feed into model
        This function allows to precompute useful tables that will be used by the model
        """

        sales_danone = self.sales_danone.copy()

        # 1. Formatting DC data
        sales_danone_dc = format_label_data(sales_danone, 'dc')

        # 2. Selecting relevant columns
        sales_danone_dc = sales_danone_dc[
            ['calendar_yearmonth', 'sku', 'brand', 'label', 'offtake',
             'sellin', 'sellout', 'sellout_price', 'store_distribution', 'anp']]

        self.all_sales = sales_danone_dc
        sales_danone_dc.to_pickle(os.path.join(DIR_TEST_DATA, 'test_all_sales_dc.pkl'))

    def create_all_features(self, dwp, dtp):
        """ Creates all features
        """

        # SALES

        # 1.a Creates sales feature for country-brand-tier-stage-label granularity
        grouped_offtake2 = \
            self.all_sales.groupby(['calendar_yearmonth', 'brand', 'label', 'sku'])[
                'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sku_')

        # 1.b Creates target
        target = features_target(grouped_offtake2, dwp, dtp)

        # 2. Creates sales feature for skus granularity
        # grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku'])['offtake'].sum().unstack(
        #     'calendar_yearmonth').fillna(0)
        # df2 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_stage_')

        # 3. Creates sales feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'brand'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df3 = features_amount_sales(grouped_offtake2, dwp, dtp, 'brand_')

        # SELL OUT

        # 4. Creates sellout feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'brand'])['sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df4 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellout_brand_')

        # SELL IN

        # 5. Creates sellin feature for sku-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku', 'label'])[
            'sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df5 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_sku_')

        # 6. Creates sellin feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'brand'])['sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df6 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_brand_')

        # OTHER FEATURES

        # 7. Creates price country feature for sku granularity
        df7 = create_seasonality_features(dtp)

        # 8. Creates sellout price feature for sku granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku'])['sellout_price'].mean().unstack(
            'calendar_yearmonth').fillna(0)
        df8 = features_amount_sales(grouped_offtake2, dwp, dtp, 'price_country_', timelag=5)

        # 9. Creates anp feature for sku granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku'])['anp'].mean().unstack(
            'calendar_yearmonth').fillna(0)
        df9 = features_amount_sales(grouped_offtake2, dwp, dtp, 'anp_', timelag=2)

        # 10. Creates store distribution feature for sku granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku'])['store_distribution'].mean().unstack(
            'calendar_yearmonth').fillna(0)
        df10 = features_amount_sales(grouped_offtake2, dwp, dtp, 'store_distribution_', timelag=2)

        dfs = [df1,
               # df2,
               # df3,
               # df4,
               df5,
               # df6,
               df7, df8, df9, df10, target]

        # Merging features
        dffinal = reduce(lambda left, right: pd.merge(left, right, on=list(
            {'date_when_predicting', 'date_to_predict', 'country', 'brand', 'tier', 'stage', 'label',
             'sku'} & set(right.columns)), how='left'), dfs)

        # one_hot = pd.get_dummies(dffinal['sku'])
        # dffinal = pd.concat([dffinal, one_hot], axis=1)

        return dffinal

    def correct_fc(self, res, month_to_correct=(7, 'CNY', 11), thrsh=0.05):
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
        tempdc = temp[temp.label == 'dc']

        for y in years:
            for m in month_to_correct:
                # Post-process for dc
                tempdc = apply_forecast_correction(sales=sales, forecast=temp, forecast_filtered=tempdc, label='dc',
                                                   year=y, month=m, thrsh=thrsh)

        return tempdc

    def forecast_since_date_at_horizon(self, date_start, horizon, model_config=None):
        """ Function that performs a full forecast since a date of sales

        :param date_start: last date of available sales in the data that need to be used by the model when forecasting
        :param horizon: horizon in the future for which we want a forecast
        :param model_config: instance of ModelConfig that contains information about the model that will be used
        :return: a dataframe containing the forecast
        """

        if model_config is None:
            print("model not specified - using default model")
            model_config = deepcopy(self.default_model_config)
            print(model_config)

        max_date_available = self.all_sales.calendar_yearmonth.max()
        filter_date = min(date_start, max_date_available)
        dwps = create_list_period(201701, filter_date, False)
        dwp, dtp = get_all_combination_date(dwps, horizon)

        print("creating the main table")
        table_all_features = self.create_all_features(dwp, dtp)
        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        features_int = ["date_when_predicting", "label", "date_to_predict", "sku", "target", "country", "brand",
                        "tier", "stage", "horizon"]
        features = [x for x in table_all_features.keys() if x not in features_int]
        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()

        for h in range(1, horizon + 1):
            print("training model at horizon: " + str(h))
            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())]
            x_train = subdata[features].values
            y_train = subdata.target
            data_test = table_all_features[(table_all_features.date_when_predicting == filter_date) &
                                           (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values
            model = MLDCModel(
                model_name=model_config.model_name,
                model_params=model_config.model_params
            )

            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            preds = preds.clip(min=0)

            data_test['prediction'] = preds
            data_test['horizon'] = h
            res = pd.concat([res, data_test[["label", "date_to_predict", "sku", "horizon", "prediction"]]])

            # Creating feature importance
            feature_importance = dict(zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(), ignore_index=True)

        self.feature_importance = feature_importance_df
        self.feature_importance.to_csv(str(date_start)+'_feature_importance.csv''')
        # Applying postprocessing
        res.to_pickle(os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_dc.pkl'))
        res = self.correct_fc(res, month_to_correct=[7, 'CNY', 11])

        return res

    def recreate_past_forecasts(self, table_all_features, list_dwps, horizon=10, model_config=None):

        if model_config is None:
            print("model not specified - using default model")
            model_config = deepcopy(self.default_model_config)
            print(model_config)

        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        features_int = ["date_when_predicting", "label", "date_to_predict", "sku", "target", "country", "brand",
                        "tier", "stage", "horizon"]
        features = [x for x in table_all_features.keys() if x not in features_int]

        resfinal = pd.DataFrame()
        feature_importance_df_final = pd.DataFrame()
        for datwep in list_dwps:
            print("date when predicting: " + str(datwep))
            res = pd.DataFrame()
            feature_importance_df = pd.DataFrame()
            for h in range(1, horizon + 1):
                print("training model at horizon: " + str(h))
                subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                             & (table_all_features.date_to_predict <= datwep)]
                x_train = subdata[features].values
                y_train = subdata.target

                data_test = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                               (table_all_features.horizon == h)].copy()

                x_test = data_test[features].values
                model = MLDCModel(
                    model_name=model_config.model_name,
                    model_params=model_config.model_params
                )

                model.fit(x_train, y_train)
                preds = model.predict(x_test)
                preds = preds.clip(min=0)
                data_test['horizon'] = h
                data_test['prediction'] = preds
                res = pd.concat([res, data_test[["label", "date_to_predict", "sku", "prediction", "horizon"]]])

                feature_importance = dict(
                    zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
                feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
                feature_importance_df = feature_importance_df.append(feature_importance.reset_index(), ignore_index=True)
            feature_importance_df['date_when_preidct'] = datwep
            self.feature_importance = feature_importance_df
            self.feature_importance.to_csv(str(datwep) + '_feature_importance.csv''')
            res = self.correct_fc(res, month_to_correct=[7, 'CNY', 11], thrsh=0.05)

            resfinal = pd.concat([resfinal, res])
            feature_importance_df_final = pd.concat([feature_importance_df_final,feature_importance_df])
            feature_importance_df_final.to_csv('./data/feature_importance_all_df.csv')

        return resfinal


if __name__ == '__main__':

    import src.forecaster.utilitaires as util
    import src.forecaster.diagnostic as diagnostic

    raw_master = pd.read_csv('./data/raw/raw_master_dc_20191126.csv')
    mod = Modeldc(raw_master)
    max_date_available = mod.all_sales.calendar_yearmonth.max()
    filter_date = min(201909, max_date_available)
    dwps = util.create_list_period(201701, filter_date, False)
    dwp, dtp = util.get_all_combination_date(dwps, 12)

    print("creating the main table")
    table_all_features = mod.create_all_features(dwp, dtp)
    # table_all_features = pd.read_csv("data/table_all_features_dc.csv")

    dwp_test = util.create_list_period(201804, 201909, False)
    #
    # model_config = ModelConfig(
    #     model_name="GradientBoostingRegressor",
    #     model_params={
    #         'standard_scaling': False,
    #         'pca': 0,
    #         'loss': 'huber',
    #         'learning_rate': 0.01,
    #         'n_estimators': 500,
    #         'subsample': 0.3,
    #         'max_depth': 10,
    #         'max_features': 0.7,
    #         'alpha': 0.9,
    #     }
    # )
    #
    res = mod.recreate_past_forecasts(table_all_features, dwp_test, horizon=10)
    res.to_csv('./data/res_dc_20191126_par.csv')
    res = pd.read_csv('./data/res_dc_20191126_par.csv')
    test = diagnostic.Diagnostic(cvr=res, raw_master=raw_master,
                                 postprocess='indep', di_eib_il_format=False)
    test.cvr.to_csv('./data/cvr_dc_20191126_sku_cat.csv')
    temp = test.run_test_dc(plot=False, horizon=7)



    print(test)