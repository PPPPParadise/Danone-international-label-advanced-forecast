import os
import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from functools import reduce
from src.forecaster.features import features_target_r6m, features_amount_sales, \
    create_seasonality_features
from src.forecaster.model import Model
from cfg.paths import DIR_TEST_DATA
from src.forecaster.utilitaires import format_label_data


class R6MModel(Model):

    def __init__(self, raw_master, *args, **kwargs):
        super(R6MModel, self).__init__(data=raw_master, *args, **kwargs)

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
        all_sales = all_sales[[
            'calendar_yearmonth', 'sku_wo_pkg', 'country', 'brand', 'tier',
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
        target = features_target_r6m(grouped_offtake2, dwp)

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

    def arima_pqd(self, df, p, q, d):
        X = df["last_6_months"].values
        size = int(len(X) * 0.3)
        train, test = X[0:size], X[size:len(X)]
        history = [x for x in train]
        predictions = list()
        predictions_ix = list()

        for ix, y in enumerate(test):
            model = ARIMA(history, order=(p, q, d))
            model_fit = model.fit(disp=0)
            # print(model_fit.summary())

            # Forecast what will happen 5 months from now
            output, _, _ = model_fit.forecast(5)
            yhat = output[-1]
            predictions.append(yhat)
            predictions_ix.append(size + ix)

            # observed value is 5 months in the future
            # obs = test[ix + 5]
            obs = df['target_r6m'].values[size + ix]

            # update history
            history.append(test[ix])
            print('predicted=%f, expected=%f' % (yhat, obs))

        error = mean_squared_error(
            df['target_r6m'].values[predictions_ix],
            predictions
        )
        print('Test MSE: %.3f' % error)

        return predictions, error

    def forecast(self, table_all_features, label, filter_date=None):
        # filter date is here the latest date until we want the r6m correction to be applied
        # if none the correction is applied while there is data
        if not filter_date:
            filter_date = 210010
        df = (
            table_all_features
                .query(f"label == '{label}'")
                .query("horizon == 10")
                .query(f"date_to_predict <= {filter_date}")
                .groupby(['date_when_predicting', 'date_to_predict'], as_index=False)
                .agg({
                    'horizon': 'mean',
                    'target_r6m': 'sum',
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
                .assign(
                date_when_predicting=lambda x: pd.to_datetime(
                    x['date_when_predicting'].apply(str), format="%Y%m"),
                date_to_predict=lambda x: pd.to_datetime(
                    x['date_to_predict'].apply(str), format="%Y%m"),
                last_6_months=lambda x: x['label_sales1'] + x['label_sales2'] + x[
                    'label_sales3'] + x['label_sales4'] + x['label_sales5'] + x[
                                            'label_sales6'],
            )
        )

        # params arima
        params_arima = {
            "il": (0, 0, 0),
            "eib": (2, 1, 0),
            "di": (2, 2, 0),
        }

        # Choose best pqd
        (p, q, d) = params_arima[label]
        predictions, _ = self.arima_pqd(df, p, q, d)
        df['predictions'] = [np.nan] * int(len(df) * 0.3) + predictions
        df = (
            df
            .assign(
                label=label,
                date_to_predict=lambda x: x["date_to_predict"].dt.strftime("%Y%m").astype(int),
                date_when_predicting=lambda x: x["date_when_predicting"].dt.strftime("%Y%m").astype(int),
            )
            .filter(["label", "date_when_predicting", "date_to_predict", "predictions"])
            .rename(columns={"predictions": "r6m_predictions"})
        )

        return df

    def correct_fc(self, res, month_to_correct, thrsh):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return:
        """
        pass

    def forecast_since_date_at_horizon(self, date_start, horizon, params):
        pass

    def recreate_past_forecasts(self, list_dwps, params, horizon):
        pass
