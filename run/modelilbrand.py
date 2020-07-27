#!/usr/bin/env python
# coding: utf-8

import sys,os
# sys.path.append(os.path.dirname(os.getcwd()))
from functools import reduce
# import xgboost as xgb
from src.exploration.MLDIModel import GlobalDIModel
from cfg.paths import *
from src.forecaster.features import *
from src.forecaster.model import Model
from src.forecaster.modelr6m import R6MModel
from src.forecaster.utilitaires import *
from src.forecaster.MLDCModel import MLDCModel
import warnings 
warnings.filterwarnings('ignore')



class Modelilbrand(Model):
    """
    XGB Regressor model with postprocessing of the output
    One XGB model is trained for each prediction horizon
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug = False
        self.config = config
        if self.debug:
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")
            print("       DEBUG       ")
            print("⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠⚠️")
    
    def Load_raw_master(self):
        """ Function to load raw master and group value by brand level
        """
        
        df =  self.raw_master
        df['date'] = pd.to_datetime(df['date'])
        df['calendar_yearmonth'] = pd.to_datetime(df['date']).dt.year.astype(
                    str) + pd.to_datetime(df['date']).dt.month.astype(str).str.zfill(2)
        df['calendar_yearmonth'] = df['calendar_yearmonth'].astype(int)
        df['brand_offtake_il'] = df.groupby(['date','country_brand'])['offtake_il'].transform(sum)
        df['brand_offtake_di'] = df.groupby(['date','country_brand'])['offtake_di'].transform(sum)
        df['brand_offtake_eib'] = df.groupby(['date','country_brand'])['offtake_eib'].transform(sum)
        df['brand_sellin_eib'] = df.groupby(['date','country_brand'])['sellin_eib'].transform(sum)
        df['brand_sellin_il'] = df.groupby(['date','country_brand'])['sellin_il'].transform(sum)
        df['brand_sellin_di'] = df.groupby(['date','country_brand'])['sellin_di'].transform(sum)
        df['brand_sellout'] = df.groupby(['date','country_brand'])['sellout'].transform(sum)
        df['brand_price'] = df.groupby(['date','country_brand'])['price'].transform(np.mean)
        df['uprep'] = df.groupby(['date','country_brand'])['uprep'].transform(np.mean)
        df['upre'] = df.groupby(['date','country_brand'])['upre'].transform(np.mean)
        df['spre'] = df.groupby(['date','country_brand'])['spre'].transform(np.mean)
        df['mainstream'] = df.groupby(['date','country_brand'])['mainstream'].transform(np.mean)

        brands = [c for c in df.columns if 'brand' in c] 

        sel_col = ['country', 'brand', 'date', 'total_vol',
                   '0to6_month_population', 
                   '6to12_month_population',
                   '12to36_month_population', 
                   'uprep', 'upre', 'spre', 'mainstream', 
                   'country_brand','calendar_yearmonth',
                   'brand_price',
                   'brand_offtake_il', 
                   'brand_offtake_di',
                   'brand_offtake_eib',
                   'brand_sellin_eib', 
                   'brand_sellout',
                   'brand_sellin_di', 
                   'brand_sellin_il'] 

        df_ = df.drop_duplicates(['date', 'country_brand'])
        df_ = df_[sel_col]
        return df_
        
    def preformat_table(self,df_):
        """ This function allows to precompute useful tables that will be used by the model
        :param df_: dataframe after grouping by brand
        :return: DataFrame after formatting for label
        """
        
        # 1. Formatting DI data
        sales_danone_di = format_label_data(df_, 'di')

        # 2. Formatting EIB data
        sales_danone_eib = format_label_data(df_, 'eib')

        # 3. Formatting IL data
        sales_danone_il = format_label_data(df_, 'il')

        # 4. Merging data
        all_sales = pd.concat([sales_danone_eib, sales_danone_di, sales_danone_il])

        # 5. Selecting relevant columns
        all_sales = all_sales[
            ['calendar_yearmonth', 'country_brand', 'country', 'brand', 'label',
             'offtake', 'sellin','sellout','price','upre','spre','mainstream']]
        
        return all_sales
        
    def Format_label(self):
        """ This function used to execute load master function and perforamt function
        :return: DataFrame after formatting for label
        """
        
        df = self.Load_raw_master()
        df.rename(columns = {'brand_offtake_il':'offtake_il',
                             'brand_offtake_di':'offtake_di',
                             'brand_offtake_eib':'offtake_eib',
                             'brand_sellin_eib':'sellin_eib',
                             'brand_sellout':'sellout', 
                             'brand_sellin_di':'sellin_di',
                             'brand_sellin_il':'sellin_il',
                             'brand_price':'price'}, inplace = True)
        df = self.preformat_table(df) 
        self.all_sales = df.reset_index(drop = True)
        return df.reset_index(drop = True)

    def Create_feature(self, dwp, dtp):
        """ The function used to Creates all features
        :param dwp: date list when predicting
        :param dtp: date list to predict
        :return: DataFrame with all features related to offtake,sellin and category.
        """
    
        df = self.Format_label()
    
        grouped_offtake2 =  df.groupby(['calendar_yearmonth','label','country','brand', 'country_brand'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_offtake_') 
        target = features_target(grouped_offtake2, dwp, dtp)
        grouped_offtake2 = df.groupby(['calendar_yearmonth', 'label','country'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0) 
        df2 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_offtake_') 
        grouped_offtake2 = df.groupby(['calendar_yearmonth', 'label','brand'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df3 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_brand_offtake_') 
        grouped_offtake2 = df.groupby(['calendar_yearmonth', 'label'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df4 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_offtake_') 
        grouped_offtake2 = df.groupby(['calendar_yearmonth', 'country'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df5 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_offtake_') 
        grouped_offtake2 = df.groupby(['calendar_yearmonth', 'country'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df6 = features_amount_sales(grouped_offtake2, dwp, dtp, 'brand_offtake_')  

        grouped_offtake2 = df.groupby(['calendar_yearmonth','label','country','brand', 'country_brand'])[
                        'sellin'].sum().unstack('calendar_yearmonth').fillna(0)
        df7 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_sellin_') 

        grouped_offtake2 = df.groupby(['calendar_yearmonth','country_brand'])[
                        'price'].mean().unstack('calendar_yearmonth').fillna(0)
        df8 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_price_', timelag=3) 

        df9 = create_seasonality_features(dtp)

        grouped_offtake2 = df.groupby(['calendar_yearmonth','country_brand'])[
                        'upre'].mean().unstack('calendar_yearmonth').fillna(0)
        df10 = features_amount_sales(grouped_offtake2, dwp, dtp, 'upre_', timelag=3) 

        grouped_offtake2 = df.groupby(['calendar_yearmonth','country_brand'])[
                        'spre'].mean().unstack('calendar_yearmonth').fillna(0)
        df11 = features_amount_sales(grouped_offtake2, dwp, dtp, 'spre_', timelag=3) 

        grouped_offtake2 = df.groupby(['calendar_yearmonth','country_brand'])[
                        'mainstream'].mean().unstack('calendar_yearmonth').fillna(0) 
        df12 = features_amount_sales(grouped_offtake2, dwp, dtp, 'mainstream_', timelag=3)

        dfs = [df1,
               df2, 
               df3,
               df4,
               df5,
               df6,
               df7,
               df8,
               df9,
               df10,
               df11,
               df12,
               target]

        # Merging features
        dffinal = reduce(lambda left, right: pd.merge(left, right, on=list(
                    {'date_when_predicting', 'date_to_predict', 'country', 'brand','label',
                     'country_brand'} & set(right.columns)), how='left'), dfs)

        # Adding onehot encoding labels
        one_hot_brand = pd.get_dummies(dffinal['country_brand'])
        one_hot_label = pd.get_dummies(dffinal['label'])
        dffinal = pd.concat([dffinal, one_hot_brand,one_hot_label], axis=1)
    
        return dffinal

    def correct_fc_il(self, res, month_to_correct=(6,'CNY', 11), thrsh=0.05):
        """ This function is a post-processing step to the forecast, changing the forecast of some month to
        correspond to past observed ratio
        :param res: the data frame containing forecasts
        :param month_to_correct: the months of forecast where we want to apply a post-process
        :param thrsh: a threshold under which we do not perform any post-processing
        :return: DataFrame after correction
        """

        sales = self.all_sales[self.all_sales.label=='il'].groupby(
            ['calendar_yearmonth', 'label','country','brand','country_brand'])['offtake'].sum().reset_index()
        temp = res.copy()

        years = list((res.date_to_predict // 100).unique())
        tempil = temp[temp.label == 'il']

        for y in years: 
            for m in month_to_correct:
                for a in tempil.country_brand.unique():
                    # Post-process for DI
                    tempil = apply_forecast_correction(sales=sales, forecast=tempil, forecast_filtered=tempil, label='il',
                                                       brand=a,year=y, month=m)
        return tempil
    
    def Load_category(self):
        """ Load category input data
        :return: DataFrame including all category features
        """
    
        category_path = self.config["data_folder_path"] + '/' + self.config["input_category_forecast"]
        df_feature_addon = pd.read_csv(category_path)

        return df_feature_addon

    def Merge_table_with_category(self,table_all_features,df_feature_addon):
        """ Merge table with all features with category calculated features
        :param table_all_features: the data frame containing all main features
        :param df_feature_addon: the data frame containing category precalculated features
        :return: DataFrame including all features including category features
        """

        table_all_features = table_all_features.merge(
            df_feature_addon[self.config["features_pop_col"] +\
                             self.config["features_cat_fsct_col"] + 
                             ['label','country_brand','date_when_predicting','date_to_predict']], 
            left_on=['label','country_brand','date_when_predicting','date_to_predict'],
            right_on =['label','country_brand','date_when_predicting','date_to_predict'])

        return table_all_features
    
    def Load_50_feature(self):
        """ Load the most important 50 features
        :return: Dataframe of the most 50 features for each horizon
        """
    
        feature_path = self.config["temp_folder_path"] + '/' + self.config["feature_import_first_run"]
        feature_importance_df_sets = pd.read_csv(feature_path, index_col = 0)

        return feature_importance_df_sets


    def forecast_since_date_at_horizon(self, date_start, horizon):
        """ Function that performs a full forecast since a date of sales
        :param date_start: last date of available sales in the data that need to be used by the model when forecasting
        :param horizon: horizon in the future for which we want a forecast
        :param params: parameters of the xgboost model
        :return: a dataframe containing the forecast
        """

        filter_date = min(date_start, self.max_date_available)
        dwps = create_list_period(201801, filter_date, False)
        dwp, dtp = get_all_combination_date(dwps, horizon)

        print("creating the main table")
        table_all_features = self.Create_feature(dwp, dtp)
        
        table_all_features['horizon'] =             (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)
        # Load category features
        df_feature_addon = self.Load_category()
        # Merge main features with category features
        table_all_features = self.Merge_table_with_category(table_all_features,
                                                            df_feature_addon)
        # Choose useful features
        features = [x for x in table_all_features.keys() if x not in self.config['features_int']]
        features_cat = [ c for c in table_all_features.keys() if ('spre' in c)|('upre' in c)|('mainstream' in c)]
        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()

        for h in range(1, horizon + 1):
            print("training model at horizon: " + str(h))
            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                         & (table_all_features.date_to_predict <= filter_date)]
            
            if not self.config["FirstRun"]:
                feature_importance_df_sets = self.Load_50_feature()
                features = list(feature_importance_df_sets[str(h)]) +\
                           features_cat +\
                           self.config["features_pop_col"] +\
                           self.config["features_cat_fsct_col"]
        
            x_train = subdata[features].values
            y_train = subdata.target

            data_test = table_all_features[(table_all_features.date_when_predicting == filter_date) &
                                                   (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values
            
            model = MLDCModel(
                model_name = self.config["model_config_RandomForestRegressor"].model_name,
                model_params = self.config["model_config_RandomForestRegressor"].model_params)
            
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            preds = preds.clip(min=0)
            data_test['horizon'] = h
            data_test['prediction'] = preds 
            res = pd.concat([res, data_test[["label","date_to_predict", "country","brand",
                                             "country_brand","prediction", "horizon"]]]) 
            # Create feature importance
            feature_importance = dict(zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(), ignore_index=True)

        self.feature_importance = feature_importance_df
        feature_importance_df['date_when_preidct'] = date_start
        feature_importance_df.to_csv(os.path.join(
            DIR_TEM, 'feature_importance_df_sets_' + str(date_start) + '.csv'))
        
        # Applying post-processing
        resfinal = self.correct_fc_il(res, month_to_correct=['CNY', 11], thrsh=0.05)
        resfinal["date_when_predicting"] = (
            pd.to_datetime(resfinal["date_to_predict"].astype(int).astype(str), format="%Y%m")
            - resfinal['horizon'].apply(pd.offsets.MonthBegin)
        ).apply(lambda x: x.strftime("%Y%m")).astype(int)
        
        # Output resutls
        res.to_csv((os.path.join(
            DIR_TEM, 'IL_Brand_Forecst_result' + str(date_start) + '.csv')),index = False)
#         res.to_pickle(os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_il.pkl'))
        return resfinal.reset_index(drop = True)

if __name__ == '__main__':

    raw_master = pd.read_csv('../data/raw_master_il_20200210_Update_0602_subbrand_catupdate_v2_till202005.csv')
    Modelil(raw_master).forecast_since_date_at_horizon(201901, 12)