from functools import reduce

from src.forecaster.features import *
from src.forecaster.utilitaires import *


class Model:
    """
    XGB Regressor model with postprocessing of the output
    One XGB model is trained for each prediction horizon
    """

    def __init__(self, data):
        self.sales_danone = data
        self.all_sales = None
        self.preformat_table()

    def preformat_table(self):
        """ Preformat data table to feed into model
        This function allows to precompute usefull tables that will be used by the model
        """

        # 0. Conversion into int format
        self.sales_danone['calendar_yearmonth'] = pd.to_datetime(self.sales_danone['date']).dt.year.astype(
            str) + pd.to_datetime(
            self.sales_danone['date']).dt.month.astype(str).str.zfill(2)
        self.sales_danone['calendar_yearmonth'] = self.sales_danone['calendar_yearmonth'].astype(int)

        # 1. Formatting DI data
        sales_danone_di = self.sales_danone.copy()
        sales_danone_di['label'] = 'di'
        sales_danone_di['offtake'] = sales_danone_di['offtake_di']
        sales_danone_di['sellin'] = sales_danone_di['sellin_di']

        # 2. Formatting EIB data
        sales_danone_eib = self.sales_danone.copy()
        sales_danone_eib['label'] = 'eib'
        sales_danone_eib['offtake'] = sales_danone_eib['offtake_eib']
        sales_danone_eib['sellin'] = sales_danone_di['sellin_eib']

        # 3. Formatting IL data
        sales_danone_il = self.sales_danone.copy()
        sales_danone_il['label'] = 'il'
        sales_danone_il['offtake'] = sales_danone_il['offtake_il']
        sales_danone_il['sellin'] = sales_danone_il['sellin_il']

        # 4. Merging data
        all_sales = pd.concat([sales_danone_eib, sales_danone_di, sales_danone_il])
        all_sales = all_sales[
            ['calendar_yearmonth', 'sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'label', 'offtake', 'sellin',
             'sellout']]

        self.all_sales = all_sales
        return all_sales

    def create_all_features(self, dwp, dtp):
        """ Creates all features
        """

        # SALES

        # 1.a Creates sales feature for country-brand-tier-stage-label granularity
        # self.preformat_table()
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

    def rescale_il_dieib(self, res):
        """ RESCALING: to enforce relation IL = EIB + DI
        :param res: dataframe that contains forecasts
        :return:
        """
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
