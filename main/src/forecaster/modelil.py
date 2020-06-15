from functools import reduce
import xgboost as xgb
from src.exploration.MLDIModel import GlobalDIModel

from cfg.paths import DIR_TEST_DATA
from src.forecaster.features import *
from src.forecaster.model import Model
from src.forecaster.modelr6m import R6MModel
from src.forecaster.utilitaires import *

DEFAULT_DATA_LAG_IL = 2


class Modelil(Model):
    """
    XGB Regressor model with postprocessing of the output
    One XGB model is trained for each prediction horizon
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = False
        if self.debug:
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")
            print("       DEBUG       ")
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")

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
        all_sales = all_sales[
            ['calendar_yearmonth', 'sku_wo_pkg', 'country', 'brand', 'tier',
             'stage', 'label', 'offtake', 'sellin', 'sellout']]

        self.all_sales = all_sales
        all_sales.to_pickle(os.path.join(DIR_TEST_DATA, 'test_all_sales_il.pkl'))

    def create_all_features(self, dwp, dtp):
        """ Creates all features
        """

        # SALES

        # 1.a Creates sales feature for country-brand-tier-stage-label granularity
        grouped_offtake2 = \
            self.all_sales.groupby(['calendar_yearmonth', 'country', 'brand', 'tier', 'stage', 'label', 'sku_wo_pkg'])[
                'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_stage_label_')

        # 1.b Creates target
        target = features_target(grouped_offtake2, dwp, dtp)

        # 2. Creates sales feature for country-brand-tier-stage granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df2 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_stage_')

        # 3. Creates sales feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'label'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df3 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_')

        # 4. Creates sales feature for country-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'country', 'label'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df4 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_label_')

        # 5. Creates sales feature for stage granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'stage'])['offtake'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df5 = features_amount_sales(grouped_offtake2, dwp, dtp, 'stage_')

        # SELL OUT

        # 6. Creates sellout feature for sku-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg', 'label'])[
            'sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df6 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellout_sku_label_')

        # 7. Creates sellout feature for sku granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg'])['sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df7 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellout_sku_')

        # 8. Creates sellout feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'label'])['sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df8 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellout_label_')

        # 9. Creates sellout feature for country-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'country', 'label'])['sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df9 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sell_out_country_label_')

        # 10. Creates sellout feature for stage granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'stage'])['sellout'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df10 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sell_out_stage_')

        # SELL IN

        # 11. Creates sellin feature for sku-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg', 'label'])[
            'sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df11 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_sku_label_')

        # 12. Creates sellin feature for sku granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'sku_wo_pkg'])['sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df12 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_sku_')

        # 13. Creates sellin feature for label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'label'])['sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df13 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sellin_label_')

        # 14. Creates sellin feature for country-label granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'country', 'label'])['sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df14 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sell_in_country_label_')

        # 15. Creates sellin feature for stage granularity
        grouped_offtake2 = self.all_sales.groupby(['calendar_yearmonth', 'stage'])['sellin'].sum().unstack(
            'calendar_yearmonth').fillna(0)
        df15 = features_amount_sales(grouped_offtake2, dwp, dtp, 'sell_in_stage_')

        # 16. Creates seasonality feature
        df16 = create_seasonality_features(dtp)

        dfs = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12, df13, df14, df15, df16, target]

        # Merging features
        dffinal = reduce(lambda left, right: pd.merge(left, right, on=list(
            {'date_when_predicting', 'date_to_predict', 'country', 'brand', 'tier', 'stage', 'label',
             'sku_wo_pkg'} & set(right.columns)), how='left'), dfs)

        # Adding onehot encoding labels
        one_hot = pd.get_dummies(dffinal['label'])
        dffinal = pd.concat([dffinal, one_hot], axis=1)

        return dffinal

    def correct_fc_global_il(self, res, month_to_correct=('CNY', 11), thrsh=0.05):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        It is applied only for label = "il"
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return:
        """

        sales = self.all_sales.groupby(['calendar_yearmonth', 'label'])['offtake'].sum().reset_index()
        temp = res.copy()
        years = list((res.date_to_predict // 100).unique())
        tmpil = res[res.label == "il"].copy()

        for y in years:
            for m in month_to_correct:

                # Post-process for IL
                tmpil = apply_forecast_correction(sales=sales, forecast=temp, forecast_filtered=tmpil, label='il',
                                                   year=y, month=m, thrsh=thrsh)

        return tmpil

    def correct_fc(self, res, month_to_correct=('CNY', 11), thrsh=0.05, recompute_IL=True):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return:
        """
        res = correct_invalid_skus(res, self.sales_danone)
        sales = self.all_sales.groupby(['calendar_yearmonth', 'label'])['offtake'].sum().reset_index()
        temp = res.copy()

        years = list((res.date_to_predict // 100).unique())
        tempdi = temp[temp.label == 'di']
        tempeib = temp[temp.label == 'eib']
        tempil = temp[temp.label == 'il']

        for y in years:
            for m in month_to_correct:

                # Post-process for DI
                tempdi = apply_forecast_correction(
                    sales=sales,
                    forecast=temp,
                    forecast_filtered=tempdi,
                    label='di',
                    year=y,
                    month=m,
                    thrsh=thrsh
                )

                # Post-process for EIB
                tempeib = apply_forecast_correction(
                    sales=sales,
                    forecast=temp,
                    forecast_filtered=tempeib,
                    label='eib',
                    year=y,
                    month=m,
                    thrsh=thrsh
                )

                if not recompute_IL:
                    # Post-process for IL
                    tempil = apply_forecast_correction(
                        sales=sales,
                        forecast=temp,
                        forecast_filtered=tempil,
                        label='il',
                        year=y,
                        month=m,
                        thrsh=thrsh
                    )

        if recompute_IL:
            # Recomputing IL to enforce IL = EIB + DI
            tempdi.rename(columns={'prediction': 'preddi'}, inplace=True)
            tempeib.rename(columns={'prediction': 'predeib'}, inplace=True)
            tempil = pd.merge(
                tempdi[['date_to_predict', 'sku_wo_pkg', 'preddi', 'horizon']],
                tempeib[['date_to_predict', 'sku_wo_pkg', 'predeib', 'horizon']],
                how='left',
                on=['date_to_predict', 'sku_wo_pkg', 'horizon']
            )
            tempil['prediction'] = tempil['preddi'] + tempil['predeib']

            tempil['label'] = 'il'
            tempdi.rename(columns={'preddi': 'prediction'}, inplace=True)
            tempeib.rename(columns={'predeib': 'prediction'}, inplace=True)

        return pd.concat([tempdi, tempeib, tempil], sort=True).drop(['preddi', 'predeib'], axis=1, errors="ignore")

    def rescale_il_dieib(self, res):
        """ RESCALING: to enforce relation IL = EIB + DI
        :param res: dataframe that contains forecasts
        :return:
        """
        res = correct_invalid_skus(res, self.sales_danone)
        resil = res[res.label == "il"].copy()[['date_to_predict', 'sku_wo_pkg', 'prediction']]
        reseibdi = res[~(res.label == "il")].copy()
        reseibdi = reseibdi.groupby(['date_to_predict', 'sku_wo_pkg'])['prediction'].sum().reset_index()
        reseibdi.rename(columns={'prediction': 'predzero'}, inplace=True)
        resmerged = pd.merge(resil, reseibdi, how='left', on=['date_to_predict', 'sku_wo_pkg'])
        resmerged['ratio'] = resmerged['prediction'] / resmerged['predzero']
        res = pd.merge(res, resmerged[['date_to_predict', 'sku_wo_pkg', 'ratio']],
                       on=['date_to_predict', 'sku_wo_pkg'],
                       how='left')
        res.loc[res['label'] == 'il', 'ratio'] = 1
        res['prediction'] = res.prediction * res.ratio
        return res

    def forecast_since_date_at_horizon(self, date_start, horizon, params):
        """ Function that performs a full forecast since a date of sales
        :param date_start: last date of available sales in the data that need to be used by the model when forecasting
        :param horizon: horizon in the future for which we want a forecast
        :param params: parameters of the xgboost model
        :return: a dataframe containing the forecast
        """
        params_dict_eib = params['eib']
        params_dict_di = params['di']

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
            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                         & (table_all_features.date_to_predict <= filter_date)]
            x_train = subdata[features].values
            y_train = subdata.target
            data_test_eib = table_all_features[
                (table_all_features.date_when_predicting == filter_date) &
                (table_all_features.horizon == h) &
                (table_all_features.label == 'eib')]

            data_test_di = table_all_features[
                (table_all_features.date_when_predicting == filter_date) &
                (table_all_features.horizon == h) &
                (table_all_features.label == 'di')]

            x_test_eib = data_test_eib[features].values
            x_test_di = data_test_di[features].values

            PARAMS = params_dict_eib[h]
            # Instantiate model
            model = xgb.XGBRegressor(**PARAMS)
            model.fit(x_train, y_train)
            preds_eib = model.predict(x_test_eib)

            PARAMS = params_dict_di[h]
            # Instantiate model
            model = xgb.XGBRegressor(**PARAMS)
            model.fit(x_train, y_train)
            preds_di = model.predict(x_test_di)

            preds_eib = preds_eib.clip(min=0)
            preds_di = preds_di.clip(0)

            data_test_eib = data_test_eib.assign(
                horizon=h,
                prediction=preds_eib
            )
            data_test_di = data_test_di.assign(
                horizon=h,
                prediction=preds_di
            )

            data_test = pd.concat([data_test_eib, data_test_di])
            res = pd.concat([
                res,
                data_test[["label", "date_to_predict", "sku_wo_pkg", "prediction",
                 "date_when_predicting", "horizon"]]
            ])

            # Creating feature importance
            feature_importance = dict(zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(), ignore_index=True)

        self.feature_importance = feature_importance_df

        res.to_pickle(os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_il.pkl'))


        # Applying post-processing
        resfinal = self.correct_fc(res, month_to_correct=['CNY', 11], thrsh=0.05, recompute_IL=True)

        cols = ["sku_wo_pkg", "date_to_predict", "horizon", "date_when_predicting",
                "label", 'prediction']


        resfinal = self.rescale_with_r6m_predictions(
            resfinal,
            labels=["eib"],
            filter_date=202010
        ).filter(cols)

        resfinal["date_when_predicting"] = (
            pd.to_datetime(resfinal["date_to_predict"].astype(int).astype(str), format="%Y%m")
            - resfinal['horizon'].apply(pd.offsets.MonthBegin)
        ).apply(lambda x: x.strftime("%Y%m")).astype(int)

        return resfinal

    def predict_global_il(self, table_all_features, dwp, horizon):
        """
        Produces predictions at a given date_when_predicting and horizon at global
        level for il market
        :param table_all_features: feature table as created by `create_all_features`
        :param dwp: int - date when predicting. example: 201807
        :param horizon: int - number of months ahead we are predicting
        :return:
        """

        df = table_all_features.copy()

        # Keep only il
        df = df[df['il'] == 1]
        df = df.drop(['di', 'eib', 'il'], axis=1)

        # Aggregate features at global level
        df = (
            df
            .groupby(['date_when_predicting', 'date_to_predict'], as_index=False)
            .agg({
                'horizon': 'mean',
                'target': 'sum',
                'month': 'mean',
                'sin_month': 'mean',
                'cos_month': 'mean',
                'label_sales1': 'mean',
                'label_sales2': 'mean',
                'label_sales3': 'mean',
                'label_sales4': 'mean',
                'label_sales5': 'mean',
                'label_sales6': 'mean',
                'label_sales7': 'mean',
                'label_sales8': 'mean',
                'label_sales9': 'mean',
                'label_sales10': 'mean',
                'label_sales11': 'mean',
            })
        )

        # create list of features
        features_int = ["date_when_predicting", "label", "date_to_predict",
                        "sku_wo_pkg", "target", "country", "brand",
                        "tier", "stage", "horizon"]
        features = [x for x in df.keys() if x not in features_int]

        # filter on training data
        train_data = (
            df
            .query(f'horizon == {horizon} and date_to_predict <= {dwp}')
        )

        # filter test_data
        test_date = (
            df
            .query(f"horizon == {horizon} and date_when_predicting == {dwp}")
        )

        # Get numpy arrays
        x_train, y_train = train_data[features].values, train_data.target.values
        x_test, y_test = test_date[features].values, test_date.target.values

        # Instantiate model
        model = GlobalDIModel(
            horizon=horizon,
            date_when_predicting=dwp,
            model_params=dict(n_jobs=-1, n_estimators=100, max_depth=5, random_state=42)
        )

        if self.debug:
            y_pred = np.ones(len(x_test)) * 5000
        else:
            # train model
            model.fit(x_train, y_train)
            # predict with model
            y_pred = model.predict(x_test)

        # format result and return it
        out = (
            test_date
            .copy()
            .assign(prediction=y_pred, label='il')
        )

        cols_to_keep = [
            'date_when_predicting',
            'date_to_predict',
            'horizon',
            'label',
            'target',
            'prediction'
        ]

        return out[cols_to_keep]

    def rescale_eib_global(self, res, res_il_global):
        """
        Rescales the sku predictions to get the same total as the global il prediction
        :param res: predictions at sku granularity
        :param res_il_global: predictions on IL at global granularity
        :return: predictions at sku granularity
        """
        print("Rescaling: IL is the sum of DI and EIB at global level ")

        # Calculer la pred DI global
        pred_di_global = (
            res
            .query("label == 'di'")
            .groupby(['date_to_predict', 'horizon'], as_index=False)
            .agg({"prediction": sum})
            .rename(columns={"prediction": "pred_di_global"})
        )

        # calculer EIB global
        pred_eib_global = (
            res_il_global
            .filter(['date_to_predict', 'horizon', 'prediction'])
            .rename(columns={"prediction": "pred_il_global"})
            .merge(
                pred_di_global,
                on=['date_to_predict', 'horizon'],
                how='outer'
            )
            .fillna(0)
            .assign(pred_eib_global=lambda x: x["pred_il_global"] - x["pred_di_global"])
            .filter(["date_to_predict", "horizon", "pred_eib_global"])
        )

        # Calculer EIB par sku
        ratio = (
            res
            .query("label == 'eib'")
            .groupby(["date_to_predict", "horizon"], as_index=False)
            .agg({"prediction": sum})
            .rename(columns={"prediction": "pred_agg_eib"})
            .merge(pred_eib_global, on=["date_to_predict", "horizon"], how='left')
            .assign(ratio=lambda x: x["pred_eib_global"] / x["pred_agg_eib"])
            .filter(["date_to_predict", "horizon", "ratio"])
        )

        pred_eib_sku = (
            res
            .query("label == 'eib'")
            .merge(ratio, on=["date_to_predict", "horizon"], how="left")
            .fillna(1)
            .assign(prediction=lambda x: x['prediction'] * x['ratio'])
            .filter(["sku_wo_pkg", "date_to_predict", "horizon", "label", "prediction", "ratio"])
        )

        # get di par sku
        pred_di_sku = (
            res
            .query("label == 'di'")
            .assign(ratio=1)
            .filter(["sku_wo_pkg", "date_to_predict", "horizon", "label", "prediction", "ratio"])
        )

        # calculer IL par sku
        pred_il_sku = (
            res
            .query("label == 'il'")
            .filter(["sku_wo_pkg", "date_to_predict", "horizon"])
            .merge(
                pred_di_sku
                .filter(["sku_wo_pkg", "date_to_predict", "horizon", "prediction"])
                .rename(columns={"prediction": "pred_di_sku"}),
                on=["sku_wo_pkg", "date_to_predict", "horizon"],
                how='left'
            )
            .merge(
                pred_eib_sku
                .filter(["sku_wo_pkg", "date_to_predict", "horizon", "prediction"])
                .rename(columns={"prediction": "pred_eib_sku"}),
                on=["sku_wo_pkg", "date_to_predict", "horizon"],
                how='left'
            )
            .fillna(0)
            .assign(
                prediction=lambda x: x["pred_eib_sku"] + x["pred_di_sku"],
                ratio=1
            )
        )

        # 5. creer la table de prediction
        # "date_to_predict", "horizon", "label", "prediction", "ratio", "sku_wo_pkg"
        cols = ["date_to_predict", "horizon", "label", "prediction", "ratio", "sku_wo_pkg"]
        out = pd.concat([
            pred_il_sku.assign(label="il").filter(cols),
            pred_di_sku.assign(label="di").filter(cols),
            pred_eib_sku.assign(label="eib").filter(cols),
        ])

        return out

    def rescale_with_r6m_predictions(self, resfinal, filter_date, labels: list):
        """
        Rescales the sku predictions using predictions from the a model predicting directly
        the cumulative sales between t+5 and t+10
        :param resfinal:
        :param filter_date: last avaiable date. int. example : 201907
        :param labels:
        :return:
        """

        # check correct labels are supplied
        supported_labels = {"il", "eib", "di"}
        assert set(labels).issubset(supported_labels), \
            f"labels {set(labels) - supported_labels} not supported. " \
            f"Only supported labels are {supported_labels}"

        # 0. Create features
        dwps = create_list_period(201601, filter_date, False)
        dwp, dtp = get_all_combination_date(dwps, 10)
        model_r6m = R6MModel(raw_master=self.raw_master)
        table_all_features = model_r6m.create_all_features(dwp=dwp, dtp=dtp)

        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting,
                            format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict,
                            format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting,
                            format='%Y%m').dt.month)

        forecasts = pd.concat([
            model_r6m.forecast(table_all_features, label, filter_date=filter_date)
            for label in labels
        ])

        # 2. Rescale

        r6m_forecasts = pd.concat([
            (
                forecasts
                .filter(["label", "date_when_predicting", "r6m_predictions"])
                .assign(horizon=h)
                .dropna()
            )
            for h in range(5, 11)  # r6m forecast is applied to horizons between 5 and 10
        ])

        ratio = (
            resfinal
            .groupby(
                ["date_to_predict", "date_when_predicting", "horizon", "label"],
                as_index=False
            )
            .agg({"prediction": sum})
            .merge(r6m_forecasts, on=["label", "date_when_predicting", "horizon"], how="left")
            .assign(ratio=lambda x: (x["r6m_predictions"] / 6) / x["prediction"])  # divide by 6 because 6 months pred
            .assign(ratio=lambda x: x["ratio"].fillna(1))
            .filter(["date_to_predict", "date_when_predicting", "horizon", "label", "ratio"])
        )

        resfinal = (
            resfinal
            .merge(ratio, on=["label", "date_when_predicting", "date_to_predict", "horizon"], how="left")
            .assign(ratio=lambda x: x["ratio"].fillna(1))
            .assign(prediction=lambda x: x["prediction"] * x["ratio"])
        )

        return resfinal

    def recreate_past_forecasts(self, table_all_features, list_dwps, params,
                                max_available_date, horizon=10):

        params_dict_eib = params['eib']
        params_dict_di = params['di']
        table_all_features['horizon'] = \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        features_int = ["date_when_predicting", "label", "date_to_predict", "sku_wo_pkg", "target", "country", "brand",
                        "tier", "stage", "horizon"]
        features = [x for x in table_all_features.keys() if x not in features_int]

        # 1. get prediction at global level for IL
        res_il_global = []
        # print("Training global model for IL")
        for datwep in list_dwps:
            # print(f".. date when predicting {datwep}")
            for h in range(1, horizon + 1):
                # print(f".... horizon {h}")
                res_il = self.predict_global_il(
                    table_all_features=table_all_features,
                    dwp=datwep,
                    horizon=h
                )

                res_il_global.append(res_il)

        res_il_global = pd.concat(res_il_global)

        # 2. correct forecasts
        n_rows = len(res_il_global)
        res_il_global = self.correct_fc_global_il(res_il_global, month_to_correct=['CNY', 11], thrsh=0.05)
        assert len(res_il_global) == n_rows, "Dropped rows when correcting predictions"

        # 3. Prediction at sku granularity di and ieb
        # print("Training SKU model for IEB and DI")
        resfinal = pd.DataFrame()
        feature_importance_df = pd.DataFrame()
        for datwep in list_dwps:
            # print(f".. date when predicting {datwep}")
            res = pd.DataFrame()
            for h in range(1, horizon + 1):
                # print(f".... horizon {h}")
                subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                             & (table_all_features.date_to_predict <= datwep)]
                x_train = subdata[features].values
                y_train = subdata.target
                data_test_eib = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                                   (table_all_features.horizon == h) &
                                                   (table_all_features.label == 'eib')]

                data_test_di = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                                   (table_all_features.horizon == h) &
                                                   (table_all_features.label == 'di')]

                x_test_eib = data_test_eib[features].values
                x_test_di = data_test_di[features].values

                if not self.debug:
                    PARAMS = params_dict_eib[h]
                    # Instantiate model
                    model = xgb.XGBRegressor(**PARAMS)
                    model.fit(x_train, y_train)
                    preds_eib = model.predict(x_test_eib)

                    PARAMS = params_dict_di[h]
                    # Instantiate model
                    model = xgb.XGBRegressor(**PARAMS)
                    model.fit(x_train, y_train)
                    preds_di = model.predict(x_test_di)

                else:
                    # dummy prediction
                    preds_di = np.ones(len(x_test_di)) * 3000
                    preds_eib = np.ones(len(x_test_eib)) * 3000

                preds_eib = preds_eib.clip(min=0)
                preds_di = preds_di.clip(0)
                data_test_eib = data_test_eib.assign(
                    horizon=h,
                    prediction=preds_eib
                )
                data_test_di = data_test_di.assign(
                    horizon=h,
                    prediction=preds_di
                )
                data_test = pd.concat([data_test_eib, data_test_di])
                res = pd.concat([res, data_test[["label", "date_to_predict", "sku_wo_pkg", "prediction", "date_when_predicting", "horizon"]]])

                feature_importance = dict(
                    zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
                feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
                feature_importance_df = feature_importance_df.append(feature_importance.reset_index(),
                                                                     ignore_index=True)

            self.feature_importance = feature_importance_df
            # rescale il di eib
            res = self.correct_fc(res, month_to_correct=['CNY', 11], thrsh=0.05, recompute_IL=True)

            resfinal = pd.concat([resfinal, res])

        resfinal["date_when_predicting"] = (
            pd.to_datetime(resfinal["date_to_predict"].astype(int).astype(str), format="%Y%m")
            - resfinal['horizon'].apply(pd.offsets.MonthBegin)
        ).apply(lambda x: x.strftime("%Y%m")).astype(int)

        orig_resfinal = resfinal.copy()
        cols = ["sku_wo_pkg", "date_to_predict", "horizon", "date_when_predicting", "label", 'prediction']

        orig_resfinal = self.rescale_with_r6m_predictions(
            orig_resfinal,
            labels=["eib"],
            filter_date=202010
        ).filter(cols)

        resfinal_scaled = self.rescale_eib_global(resfinal, res_il_global)

        # switch model
        resfinal_scaled["date_when_predicting"] = (
            pd.to_datetime(resfinal_scaled["date_to_predict"].astype(int).astype(str), format="%Y%m")
            - resfinal_scaled['horizon'].apply(pd.offsets.MonthBegin)
        ).apply(lambda x: x.strftime("%Y%m")).astype(int)

        resfinal = pd.concat([
            resfinal_scaled.filter(cols).query("date_when_predicting <= 201804"),
            orig_resfinal.filter(cols).query("date_when_predicting > 201804")
        ]).assign(ratio=1)

        return resfinal


if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import src.forecaster.utilitaires as util

    raw_master = pd.read_csv('./data/raw/raw_master_il_1115.csv')
    mod = Modelil(raw_master)
    dwp_test = util.create_list_period(201707, 201902, False)

    max_date_available = mod.all_sales.calendar_yearmonth.max()
    filter_date = min(201908, max_date_available)
    dwps = util.create_list_period(201601, filter_date, False)
    dwp, dtp = util.get_all_combination_date(dwps, 10)

    # 1. Read precalculated features
    print("creating the main table")
    table_all_features = mod.create_all_features(dwp, dtp)
    # table_all_features = pd.read_csv('./data/table_all_features.csv')

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
    res = mod.recreate_past_forecasts(table_all_features, dwp_test, dico_params, max_available_date=filter_date)
