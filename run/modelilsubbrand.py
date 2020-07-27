#!/usr/bin/env python
# coding: utf-8

import sys,os
# sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd
import numpy as np
from cfg.paths import *
import lunarcalendar 
from functools import reduce
from collections import namedtuple
from dateutil.relativedelta import relativedelta
from src.forecaster.features import *
from src.forecaster.utilitaires import *
from collections import namedtuple
from src.forecaster.MLDCModel import MLDCModel
from sklearn.model_selection import train_test_split,GridSearchCV
from src.forecaster.model import Model
from src.data_wrangling.FeatureMaster import FeatureMaster
import warnings  
ModelConfig = namedtuple("ModelConfig", "model_name model_params")
warnings.filterwarnings('ignore')


class Modelilsubbrand(Model):
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
        df['country_brand'] = df.apply( lambda x:x['country']+'_'+x['brand'], axis = 1)
        df['sub_brand'] = df.apply( lambda x:x['country']+'_'+x['brand']+'_'+x['tier'], axis = 1)
        df['calendar_yearmonth'] = pd.to_datetime(df['date']).dt.year.astype(
                    str) + pd.to_datetime(df['date']).dt.month.astype(str).str.zfill(2)
        df['calendar_yearmonth'] = df['calendar_yearmonth'].astype(int)
        df['sub_brand_offtake_il'] = df.groupby(['date','sub_brand'])['offtake_il'].transform(sum)
        df['sub_brand_offtake_di'] = df.groupby(['date','sub_brand'])['offtake_di'].transform(sum)
        df['sub_brand_offtake_eib'] = df.groupby(['date','sub_brand'])['offtake_eib'].transform(sum)
        df['sub_brand_sellin_eib'] = df.groupby(['date','sub_brand'])['sellin_eib'].transform(sum)
        df['sub_brand_sellin_il'] = df.groupby(['date','sub_brand'])['sellin_il'].transform(sum)
        df['sub_brand_sellin_di'] = df.groupby(['date','sub_brand'])['sellin_di'].transform(sum)
        df['uprep'] = df.groupby(['date','sub_brand'])['uprep'].transform(np.mean)
        df['upre'] = df.groupby(['date','sub_brand'])['upre'].transform(np.mean)
        df['spre'] = df.groupby(['date','country_brand'])['spre'].transform(np.mean)
        df['mainstream'] = df.groupby(['date','country_brand'])['mainstream'].transform(np.mean)

        brands = [c for c in df.columns if 'sub_brand' in c]  

        sel_col = ['country', 
                   'brand', 
                   'date',
                   'tier',
                   '0to6_month_population', 
                   '6to12_month_population',
                   '12to36_month_population', 
                   'uprep', 
                   'upre', 
                   'spre',
                   'mainstream', 
                   'country_brand',
                   'sub_brand',
                   'calendar_yearmonth', 
                   'sub_brand_offtake_il', 
                   'sub_brand_offtake_di',
                   'sub_brand_offtake_eib', 
                   'sub_brand_sellin_eib', 
                   'sub_brand_sellin_di', 
                   'sub_brand_sellin_il']

        df_ = df.drop_duplicates(['date', 'sub_brand'])
        df_ = df_[sel_col]
        df_['price_tier_cat']= df_.apply(lambda x:x[config_price_tier_dict[x.sub_brand]], axis = 1)  
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
            ['calendar_yearmonth', 'country_brand', 'country', 'brand','tier','sub_brand','label', 
             'offtake','sellin','upre','spre','mainstream']]
        
        return all_sales
    
    def get_rolling(self):
        """ This function is used to calculate moving average features
        :return: DataFrame with moving average features
        :return: moving average related features list
        """
        
        df = self.Load_raw_master()
        
        df_table_feature_rolling  = FeatureMaster._add_rolled_values(
            df = df, 
            date_col='date',
            granularity=['sub_brand'],
            value_cols=['sub_brand_offtake_il', 
                        'sub_brand_sellin_il',
                        'sub_brand_offtake_di',
                        'price_tier_cat'], 
            window_sizes=[3, 6, 9, 12, 24])

        df_table_feature_rolling ['date_when_predicting'] = pd.to_datetime(df_table_feature_rolling ['date']).dt.year.astype(
                    str) + pd.to_datetime(df_table_feature_rolling ['date']).dt.month.astype(str).str.zfill(2)
        df_table_feature_rolling ['date_when_predicting'] = df_table_feature_rolling ['date_when_predicting'].astype(int)
        roll_features_sel = [c for c in df_table_feature_rolling.keys() if ('mean' in c)] 
        return df_table_feature_rolling,roll_features_sel
    
    
    def Merge_table_with_rolling(self,table_all_features,df_table_feature_rolling,roll_features_sel):
        """ This function used to merge main features with rolling features
        :param table_all_features: 
        :param df_table_feature_rolling: 
        :param roll_features_sel: 
        :return: DataFrame with all features
        """
        table_all_features = table_all_features.merge(
            df_table_feature_rolling[['date_when_predicting','sub_brand']+roll_features_sel], 
            left_on =['date_when_predicting','sub_brand'],
            right_on = ['date_when_predicting','sub_brand'])
        return table_all_features
        
    def Format_label(self):
        """ This function used to execute load master function and perforamt function
        :return: DataFrame after formatting for label
        """
        df = self.Load_raw_master()
        df.rename(columns = {'sub_brand_offtake_il':'offtake_il',
                             'sub_brand_offtake_di':'offtake_di',
                             'sub_brand_offtake_eib':'offtake_eib',
                             'sub_brand_sellin_eib':'sellin_eib',
                             'sub_brand_sellout':'sellout', 
                             'sub_brand_sellin_di':'sellin_di',
                             'sub_brand_sellin_il':'sellin_il',
                             'sub_brand_price':'price'}, inplace = True)
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
    
        df = df[df.label == 'il']

        grouped_offtake2 =  df.groupby(['calendar_yearmonth','label','country','brand', 'tier','country_brand','sub_brand'])[
                        'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
        df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_sub_brand_offtake_') 
        
        target = features_target(grouped_offtake2, dwp, dtp)
        
        grouped_offtake2 = df.groupby(['calendar_yearmonth','label','country','brand', 'country_brand'])[
                        'sellin'].sum().unstack('calendar_yearmonth').fillna(0)
        df7 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_sellin_')

        df9 = create_seasonality_features(dtp)

        grouped_offtake2 = df.groupby(['calendar_yearmonth','sub_brand'])[
                        'upre'].mean().unstack('calendar_yearmonth').fillna(0)
        df10 = features_amount_sales(grouped_offtake2, dwp, dtp, 'upre_', timelag=3) 

        grouped_offtake2 = df.groupby(['calendar_yearmonth','sub_brand'])[
                        'spre'].mean().unstack('calendar_yearmonth').fillna(0)
        df11 = features_amount_sales(grouped_offtake2, dwp, dtp, 'spre_', timelag=3) 

        grouped_offtake2 = df.groupby(['calendar_yearmonth','sub_brand'])[
                        'mainstream'].mean().unstack('calendar_yearmonth').fillna(0) 
        df12 = features_amount_sales(grouped_offtake2, dwp, dtp, 'mainstream_', timelag=3)

        dfs = [df1,
               df7,
               df9,
               df10,
               df11,
               df12,
               target]

        # Merging features
        dffinal = reduce(lambda left, right: pd.merge(left, right, on=list(
                    {'date_when_predicting', 'date_to_predict', 'country', 'brand',  'label','tier',
                     'country_brand','sub_brand'} & set(right.columns)), how='left'), dfs)

        # Adding onehot encoding labels
        one_hot_brand = pd.get_dummies(dffinal['country_brand']) 
        one_hot_tier = pd.get_dummies(dffinal['tier'])
        # One_hot_label = pd.get_dummies(dffinal['label'])
        dffinal = pd.concat([dffinal, 
                             one_hot_brand,
                             one_hot_tier], axis=1)
    
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
                    # Post-process for il
                    tempil = apply_forecast_correction(sales=sales, forecast=tempil, forecast_filtered=tempil, label='il',
                                                       brand=a,year=y, month=m)
        return tempil

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
        
        table_all_features['horizon'] =  (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
            (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
             pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month)

        # Calculate rolling features
        df_table_feature_rolling,roll_features_sel = self.get_rolling()
        # Merge main features with rolling features
        table_all_features = self.Merge_table_with_rolling(table_all_features,
                                                           df_table_feature_rolling,
                                                           roll_features_sel) 
        # Choose useful features
        features = [x for x in table_all_features.keys() if (x not in self.config["features_int"])&
                    (x in self.config["feature_sub_brand"])]
        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()

        for h in range(1, horizon + 1):
            print("training model at horizon: " + str(h))
            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                         & (table_all_features.date_to_predict <= filter_date)]
        
            x_train = subdata[features].values
            y_train = subdata.target

            data_test = table_all_features[(table_all_features.date_when_predicting == filter_date) &
                                                   (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values
            
            model = MLDCModel(
                model_name = self.config["model_config_XGBRegressor_sub_brand"].model_name,
                model_params = self.config["model_config_XGBRegressor_sub_brand"].model_params)
            
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
            DIR_TEM, 'feature_importance_df_sets_sub_brand_' + str(date_start) + '.csv'))
        
        # Applying post-processing
        resfinal = self.correct_fc_il(res, month_to_correct=['CNY', 11], thrsh=0.05)
        resfinal["date_when_predicting"] = (
            pd.to_datetime(resfinal["date_to_predict"].astype(int).astype(str), format="%Y%m")
            - resfinal['horizon'].apply(pd.offsets.MonthBegin)
        ).apply(lambda x: x.strftime("%Y%m")).astype(int)
        
        # Output resutls
        res.to_csv((os.path.join(
            DIR_TEM, 'IL_sub_Brand_Forecst_result' + str(date_start) + '.csv')),index = False)
#         res.to_pickle(os.path.join(DIR_TEST_DATA, 'test_apply_forecast_correction_il.pkl'))
        return resfinal.reset_index(drop = True)


# In[3]:


# if __name__ == '__main__':

#     raw_master = pd.read_csv('../data/raw_master_il_20200210_Update_0602_subbrand_catupdate_v2_till202005.csv')
#     Modelil(raw_master).forecast_since_date_at_horizon(201901, 12)


# In[4]:


raw_master = pd.read_csv('../data/raw_master_il_20200210_Update_0602_subbrand_catupdate_v2_till202005.csv')
Sub_Modelil(raw_master).forecast_since_date_at_horizon(201901, 12)

