#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re, os, gc
import lunarcalendar
import src.kpi.kpis_computation as kpis
import src.kpi.KPI_formalization_com as kpi_stats
import src.forecaster.utilitaires as util
import src.forecaster.modelil as mod 
from scipy import stats
from functools import reduce
from collections import namedtuple
from dateutil.relativedelta import relativedelta
from src.forecaster.features import *
from src.forecaster.MLDCModel import MLDCModel
from src.forecaster.features import *
from src.forecaster.model import Model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
import warnings 
warnings.filterwarnings('ignore')


config = {}

config["data_folder_path"] = "../data"
config["temp_folder_path"] = "../temp"

#input files
config["input_raw_master"] = "raw_master_il_20200210_Update_0602_subbrand_catupdate_v2+2020.csv"
config["input_category_forecast"] = "IL_feature_table_all_0610_cat_fsct.csv"

#temp files
config["feature_import_first_run"] = "feature_importance_df_sets_0610.csv"

# Parameter configuration
config["train_start"] = 201601
config["train_end"] = 201912
config["backtest_start"] = 201801
config["backtest_end"] = 201912
config["FirstRun"] = False 
config['horizon'] = 12

# Columns configuration
config["features_int"] = ["date_when_predicting", 
                          "label",
                          "date_to_predict",
                          "target",
                          "country", 
                          "brand",
                          "horizon",
                          "country_brand_channel",
                          "country_brand"]

config["features_cat_col"] = ['0to6_month_population_mean_3M',
                              '6to12_month_population_mean_3M',
                              '12to36_month_population_mean_3M',
                              '0to6_month_population_mean_6M',
                              '6to12_month_population_mean_6M',
                              '12to36_month_population_mean_6M',
                              '0to6_month_population_mean_9M',
                              '6to12_month_population_mean_9M',
                              '12to36_month_population_mean_9M',
                              '0to6_month_population_mean_12M',
                              '6to12_month_population_mean_12M',
                              '12to36_month_population_mean_12M']   

config["features_cat_fsct_col"] = ['upre_fsct',
                                   'spre_fsct',
                                   'mainstream_fsct',
                                   'upre_mean_3M_fsct',
                                   'spre_mean_3M_fsct',
                                   'mainstream_mean_3M_fsct'] 
# Model Parameters
ModelConfig = namedtuple("ModelConfig", "model_name model_params")

config["model_config_XGBRegressor"] =  ModelConfig(
    model_name="XGBRegressor",
    model_params={
        'max_depth': 8,
        'gamma': 0.02,
        'subsample': 0.3,
        'n_estimators': 60,
        'learning_rate': 0.1,
        'n_jobs': 12,
        'verbosity': 2})

config["model_config_RandomForestRegressor"] = ModelConfig(
        model_name="RandomForestRegressor",
        model_params={
            'max_depth': 8,
            'n_estimators': 80,
            'max_features':50,
            'n_jobs': -1}) 

config["model_config_ExtraTreesRegressor"] = ModelConfig(
        model_name="ExtraTreesRegressor",
        model_params={
            'max_depth': 8,
            'n_estimators': 60,
            'max_features':50,
            'n_jobs': 12}) 

config["model_config_AdaBoostRegressor"] = ModelConfig(
        model_name="AdaBoostRegressor",
        model_params={
            'n_estimators': 80,
            'learning_rate': 0.2,
            'loss':'square'}) 

config["model_config_GradientBoostingRegressor"] = ModelConfig(
        model_name="GradientBoostingRegressor",
        model_params={ 
            'subsample': 0.3,
            'n_estimators': 80,
            'learning_rate': 0.1})


# ### Load Data


def Load_raw_master(config):
    
    raw_master_path = config["data_folder_path"] + '/' + config["input_raw_master"]
    
    df =  pd.read_csv(raw_master_path)
    df['date'] = pd.to_datetime(df['date'])
    df['country_brand'] = df.apply( lambda x:x['country']+'_'+x['brand'], axis = 1)
    df['calendar_yearmonth'] = pd.to_datetime(df['date']).dt.year.astype(
                str) + pd.to_datetime(df['date']).dt.month.astype(str).str.zfill(2)
    df['calendar_yearmonth'] = df['calendar_yearmonth'].astype(int)
    df['brand_offtake_il'] = df.groupby(['date','country_brand'])['offtake_il'].transform(sum)
    df['brand_offtake_di'] = df.groupby(['date','country_brand'])['offtake_di'].transform(sum)
    df['brand_offtake_eib'] = df.groupby(['date','country_brand'])['offtake_eib'].transform(sum)
    df['brand_sellin_eib'] = df.groupby(['date','country_brand'])['sellin_eib'].transform(sum)
    df['brand_sellin_il'] = df.groupby(['date','country_brand'])['sellin_il'].transform(sum)
    df['brand_sellin_di'] = df.groupby(['date','country_brand'])['sellin_di'].transform(sum)
    df['brand_retailer_inv'] = df.groupby(['date','country_brand'])['retailer_inv'].transform(sum)
    df['brand_sellout'] = df.groupby(['date','country_brand'])['sellout'].transform(sum)
    df['brand_sp_inv'] = df.groupby(['date','country_brand'])['sp_inv'].transform(sum)
    df['brand_price'] = df.groupby(['date','country_brand'])['price'].transform(np.mean)
    df['brand_price_krmb_per_ton_il'] = df.groupby(['date','country_brand'])['price_krmb_per_ton_il'].transform(np.mean)
    df['brand_value_krmb_il'] = df.groupby(['date','country_brand'])['value_krmb_il'].transform(np.mean)
    df['volume_ton_il'] = df.groupby(['date','country_brand'])['volume_ton_il'].transform(np.mean)
    df['uprep'] = df.groupby(['date','country_brand'])['uprep'].transform(np.mean)
    df['upre'] = df.groupby(['date','country_brand'])['upre'].transform(np.mean)
    df['spre'] = df.groupby(['date','country_brand'])['spre'].transform(np.mean)
    df['mainstream'] = df.groupby(['date','country_brand'])['mainstream'].transform(np.mean)

    brands = [c for c in df.columns if 'brand' in c] 

    sel_col = ['country', 'brand', 'date','isa', 'osa', 'total_vol',
               'if_vol', 'fo_vol', 'gum_vol', 'cl_vol', 'il_vol',
               '0to6_month_population', '6to12_month_population',
               '12to36_month_population', 'uprep', 'upre', 'spre',
               'mainstream', 'country_brand','calendar_yearmonth',
               'brand_offtake_il', 'brand_offtake_di','brand_offtake_eib',
               'brand_sellin_eib', 'brand_retailer_inv','brand_sellout',
               'brand_sp_inv', 'brand_sellin_di', 'brand_price','brand_sellin_il',
               'brand_price_krmb_per_ton_il', 'brand_value_krmb_il'] 

    df_ = df.drop_duplicates(['date', 'country_brand'])
    df_ = df_[sel_col]
    
    return df_


# In[4]:


def get_observed_sales(res: pd.DataFrame, 
                       label: str,brand: str,
                       year: int, month: int, date_name: str, value_col: str) -> float:
    """ Function to get the ratio of sales of one month compared to its surrounding months

    :param value_col:
    :param date_name:
    :param res: dataframe containing the sales
    :param label: label for which we want to compute the ratio
    :param year: year for which we want to compute the ratio
    :param month: month for which we want to compute the ratio
    :param date_name: name of the date column
    :return: the ratio
    """

    temp = res.copy()
    ob = temp[(temp[date_name] == (year * 100 + month - 1)) & (temp['country_brand'] == brand)][value_col].sum() +          temp[(temp[date_name] == (year * 100 + month + 1)) & (temp['country_brand'] == brand)][value_col].sum()
    ac = temp[(temp[date_name] == (year * 100 + month)) & (temp['country_brand'] == brand)][value_col].sum() 
 
    return ac / ob


# In[5]:


def apply_forecast_correction(sales, forecast, forecast_filtered, label,brand, year, month, thrsh=0.05):
    """ This function is used to apply the forecast correction depending upon the rule described in the documentation
    :param sales: raw_master containing the original historical sales data
    :param forecast: dataframe containing the full forecast
    :param forecast_filtered: dataframe containing forecasts corresponding to the chosen label
    :param label: chosen label
    :param year: int
    :param month: int
    :param thrsh: threshold below which the correction is not applied
    :return: dataframe containing the corrected forecast
    """

    def get_cny_month(cny_year):
        cny_date = lunarcalendar.festival.ChineseNewYear(cny_year)
        cny_month = cny_date.month

        if cny_date.day > 24:
            cny_month += 1

        return cny_month

    year_minus_1 = year - 1
    month_y1 = month
    year_minus_2 = year - 2
    month_y2 = month
    year_minus_3 = year - 3
    month_y3 = month
    year_minus_4 = year - 4
    month_y4 = month

    if month == 'CNY':
        month = get_cny_month(year)
        month_y1 = get_cny_month(year_minus_1)
        month_y2 = get_cny_month(year_minus_2)
        
    correction_condition = (int(str(year * 100 + month)) != forecast.date_to_predict.max()) &                            (int(str(year * 100 + month)) != forecast.date_to_predict.min())
    
    # When first month of forecast need adjust
    if int(str(year * 100 + month)) == forecast.date_to_predict.min():
        actual_ = sales[sales.calendar_yearmonth == int(str(year * 100 + month))-1]
        actual_['label'] = label
        actual_['brand'] = brand
        actual_['horizon'] = 0
        actual_.rename(columns = {'calendar_yearmonth':'date_to_predict',
                                  'offtake':'prediction'}, inplace = True)
        actual_ = actual_[forecast.columns]
        forecast = pd.concat([actual_, forecast])
        correction_condition = True

    if correction_condition:
        if month==11:
            tar = get_observed_sales(sales, label,brand,year_minus_4, month_y4, 
                                     'calendar_yearmonth', 'offtake')*0.1 + \
                  get_observed_sales(sales, label,brand, year_minus_3, month_y3, 
                                     'calendar_yearmonth', 'offtake')*0.2 + \
                  get_observed_sales(sales, label,brand, year_minus_2, month_y2, 
                                     'calendar_yearmonth', 'offtake')*0.3 + \
                  get_observed_sales(sales, label,brand, year_minus_1, month_y1, 
                                     'calendar_yearmonth', 'offtake')*0.4
        else:
            tar = get_observed_sales(sales, label,brand,year_minus_2, month_y2,
                                     'calendar_yearmonth', 'offtake')*0.2 + \
                  get_observed_sales(sales, label,brand, year_minus_1, month_y1,
                                     'calendar_yearmonth', 'offtake')*0.8
        acoc = get_observed_sales(forecast, 
                                  label,brand,
                                  year, month, 'date_to_predict', 'prediction')  
        mf = tar / acoc 

        if np.abs(tar - acoc) > thrsh:
            forecast_filtered.loc[(forecast_filtered.date_to_predict == year * 100 + month)&(forecast_filtered.country_brand==brand), 'prediction'] *= mf

    return forecast_filtered 


# In[6]:


def correct_fc_il(all_sales, res, month_to_correct=(6,'CNY', 11), thrsh=0.05):
    """ This function is a post-processing step to the forecast, changing the forecast of some month to
    correspond to past observed ratio
    :param res: the data frame containing forecasts
    :param month_to_correct: the months of forecast where we want to apply a post-process
    :param thrsh: a threshold under which we do not perform any post-processing
    :return:
    """

    sales = all_sales[all_sales.label=='il'].groupby(['calendar_yearmonth', 'label','country','brand','country_brand'])['offtake'].sum().reset_index()
    temp = res.copy()

    years = list((res.date_to_predict // 100).unique())
    tempil = temp[temp.label == 'il']
    tempdi = temp[temp.label == 'di']

    for y in years: 
        for m in month_to_correct:
            for a in tempil.country_brand.unique():
                # Post-process for DI
                tempil = apply_forecast_correction(sales=sales, forecast=tempil, forecast_filtered=tempil, label='il',
                                                   brand=a,year=y, month=m)
    for y in years: 
        for m in month_to_correct:
            for a in tempdi.country_brand.unique():
                # Post-process for DI
                tempdi = apply_forecast_correction(sales=sales, forecast=tempdi, forecast_filtered=tempdi, label='di',
                                                   brand=a,year=y, month=m)
    return tempil,tempdi


# In[7]:


def fill_kpi_df(ytd_forecasts_kpis, df_, level, computed_on, scopes, months, horizons, first_month=None, verbose=False):
    """
    Fills the ytd KPIs dataframe
    Inputs:
      ytd_forecasts_kpis: dataframe to be filled
      df_: the dataframe to be used for KPIs computation
      computed_on: the scope where the KPIs are computed
      scopes: list of scopes to take into account
      months: list of months where to compute KPIs
      horizons: list of horizons for KPIs computation. Can be an integer or "R6M"
    Returns:
      ytd_forecasts_kpis: the input datframe with new KPIs appended
    """
    for scope in scopes:
        for month in months:
            for horizon in horizons: 
                if horizon == "R6M":
                    r_bias = kpis.YTD_rolling_bias(6, 3, month, df_, scope=scope, agg_level="sku", verbose=verbose)
                    if r_bias is not None:
                        r_bias = 100 * r_bias 
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["bias", str(horizon), scope, level, computed_on, "ytd", month, r_bias]

                    r_bias = kpis.rolling_bias(6, 3, month, df_, scope=scope, agg_level="sku", verbose=verbose)
                    if r_bias is not None:
                        r_bias = 100 * r_bias
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["bias", str(horizon), scope, level, computed_on, "month", month, r_bias]

                else:
                    fa = kpis.YTD_FA(horizon, month, df_, scope=scope, agg_level="sku", first_month=first_month, verbose=verbose)
                    if fa is not None:
                        fa = 100 * fa
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["fa", str(horizon), scope, level, computed_on, "ytd", month, fa]

                    fa = kpis.FA(horizon, month, df_, scope=scope, agg_level="sku", verbose=verbose)
                    if fa is not None:
                        fa = 100 * fa
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["fa", str(horizon), scope, level, computed_on, "month", month, fa]


                    bias = kpis.YTD_bias(horizon, month, df_, scope=scope, agg_level="sku", first_month=first_month, verbose=verbose)
                    if bias is not None:
                        bias = 100 * bias
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["bias", str(horizon), scope, level, computed_on, "ytd", month, bias]

                    bias = kpis.bias(horizon, month, df_, scope=scope, agg_level="sku", verbose=verbose)
                    if bias is not None:
                        bias = 100 * bias
                    ytd_forecasts_kpis.loc[len(ytd_forecasts_kpis)] = ["bias", str(horizon), scope, level, computed_on, "month", month, bias]
    return ytd_forecasts_kpis


# In[8]:


def preformat_table(sales_danone):
    """ Preformat data table to feed into model
    This function allows to precompute useful tables that will be used by the model
    """

    # 1. Formatting DI data
    sales_danone_di = util.format_label_data(sales_danone, 'di')

    # 2. Formatting EIB data
    sales_danone_eib = util.format_label_data(sales_danone, 'eib')

    # 3. Formatting IL data
    sales_danone_il = util.format_label_data(sales_danone, 'il')

    # 4. Merging data
    all_sales = pd.concat([sales_danone_eib, sales_danone_di, sales_danone_il])

    # 5. Selecting relevant columns
    all_sales = all_sales[
        ['calendar_yearmonth', 'country_brand', 'country', 'brand', 'label', 'offtake', 'sellin', 'sp_inv','sellout','price','upre','spre','mainstream']]
    return all_sales 


# In[9]:


def Format_label(df):
    df.rename(columns = {'brand_offtake_il':'offtake_il',
                         'brand_offtake_di':'offtake_di',
                         'brand_offtake_eib':'offtake_eib',
                         'brand_sellin_eib':'sellin_eib',
                         'brand_retailer_inv':'retailer_inv',
                         'brand_sellout':'sellout', 
                         'brand_sp_inv':'sp_inv',
                         'brand_sellin_di':'sellin_di',
                         'brand_sellin_il':'sellin_il',
                         'brand_price':'price'}, inplace = True)
    df = preformat_table(df) 
    return df


# In[10]:


def Create_feature(df, dwp, dtp):
    
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
                {'date_when_predicting', 'date_to_predict', 'country', 'brand',  'label',
                 'country_brand'} & set(right.columns)), how='left'), dfs)

    # Adding onehot encoding labels
    one_hot_brand = pd.get_dummies(dffinal['country_brand'])
    one_hot_label = pd.get_dummies(dffinal['label'])
    dffinal = pd.concat([dffinal, one_hot_brand,one_hot_label], axis=1)
    
    return dffinal


# In[11]:


def Create_date_list(config):
    
    dwps = util.create_list_period(config["train_start"], config["train_end"], False)
    dwp_test = util.create_list_period(config["backtest_start"], config["backtest_end"], False)
    dwp, dtp = util.get_all_combination_date(dwps, 12) 
    
    return dwp_test,dwp,dtp


# In[12]:


def Load_category(config):
    
    category_path = config["data_folder_path"] + '/' + config["input_category_forecast"]
    df_feature_addon = pd.read_csv(category_path)
    
    return df_feature_addon


# In[13]:


def Calculate_horizon(table_all_features):
    
    table_all_features['horizon'] =                 (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
                 pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
                (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
                 pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month) 
    
    return table_all_features


# In[14]:


def Merge_table_with_category(table_all_features,df_feature_addon):
    
    table_all_features = table_all_features.merge(
        df_feature_addon[config["features_cat_col"] +\
                         config["features_cat_fsct_col"] + 
                         ['label','country_brand','date_when_predicting','date_to_predict']], 
        left_on=['label','country_brand','date_when_predicting','date_to_predict'],
        right_on =['label','country_brand','date_when_predicting','date_to_predict'])
    
    return table_all_features


# In[15]:


def Load_50_feature(config):
    
    feature_path = config["temp_folder_path"] + '/' + config["feature_import_first_run"]
    feature_importance_df_sets = pd.read_csv(feature_path, index_col = 0)
    
    return feature_importance_df_sets


# In[16]:


def Generate_Forecast_IL(config,dwp_test,table_all_features,df):
    resfinal = pd.DataFrame()
    feature_importance_df_final = pd.DataFrame()

    # Filter features to train the model
    features = [x for x in table_all_features.keys() if x not in config["features_int"]]
    features_cat = [ c for c in table_all_features.keys() if ('spre' in c)|('upre' in c)|('mainstream' in c)]
    

    for datwep in dwp_test: 
        print(datwep) 
        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()
        for h in range(1, config["horizon"] + 1):
            print("training model at horizon: " + str(h) )

            subdata = table_all_features[(table_all_features.horizon == h) &
                                         (~table_all_features.target.isnull())
                                                 & (table_all_features.date_to_predict <= datwep)]
            if not config["FirstRun"]:
                feature_importance_df_sets = Load_50_feature(config)
                features = list(feature_importance_df_sets[str(h)]) +\
                           features_cat +\
                           config["features_cat_col"] +\
                           config["features_cat_fsct_col"] 
                
            x_train = subdata[features].values
            y_train = subdata.target 

            data_test = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                                   (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values

#             model1 = MLDCModel(
#                         model_name = config["model_config_XGBRegressor"].model_name,
#                         model_params = config["model_config_XGBRegressor"].model_params)
            model = MLDCModel(
                        model_name = config["model_config_RandomForestRegressor"].model_name,
                        model_params = config["model_config_RandomForestRegressor"].model_params)
    #         model3 = MLDCModel(
    #                     model_name = config["model_config_GradientBoostingRegressor"].model_name,
    #                     model_params = config["model_config_GradientBoostingRegressor"].model_params)
    #         model4 = MLDCModel(
    #                     model_name = config["model_config_AdaBoostRegressor"].model_name,
    #                     model_params = config["model_config_AdaBoostRegressor"].model_params)

    #         model1.fit(x_train, y_train)
    #         model2.fit(x_train, y_train)
    #         model3.fit(x_train, y_train)
    #         model4.fit(x_train, y_train)

    #         preds1_train = model1.predict(x_train)
    #         preds2_train = model2.predict(x_train)
    #         preds3_train = model3.predict(x_train)
    #         preds4_train = model4.predict(x_train)

    #         preds1_test = model1.predict(x_test)
    #         preds2_test = model2.predict(x_test)
    #         preds3_test = model3.predict(x_test)
    #         preds4_test = model4.predict(x_test)


#             param_grid = [{
#                 'max_depth': [8],
#                 'n_estimators': [80]}]


    #             KNN = KNeighborsRegressor()
    #             grid_search = GridSearchCV(KNN, param_grid, cv=5) 
    #             grid_search.fit(np.column_stack((np.array([list(preds1_train),list(preds2_train),list(preds3_train),list(preds4_train)]).T,
    #                             pd.get_dummies(subdata['country_brand'],prefix = 'country_brand').values)), y_train)
    #             KNN_blending = grid_search.best_estimator_
    #             preds = KNN_blending.predict(np.column_stack((np.array([list(preds1_test),list(preds2_test),list(preds3_test),list(preds4_test)]).T,
    #                                          pd.get_dummies(data_test['country_brand'],prefix = 'country_brand').values)))

#             grid_search = GridSearchCV(model1.model, param_grid, cv=5) 
            model.fit(x_train, y_train)
#             model = grid_search.best_estimator_
            print (model)
            preds = model.predict(x_test)


            preds = preds.clip(min=0)
            data_test['horizon'] = h
            data_test['prediction'] = preds 
            res = pd.concat([res, data_test[["label","date_to_predict", "country","brand",
                                             "country_brand","prediction", "horizon"]]]) 
            feature_importance = dict(
                    zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(),
                                                                     ignore_index=True)
        feature_importance_df['date_when_preidct'] = datwep
        feature_importance = feature_importance_df
        feature_importance.to_csv(config["temp_folder_path"] +\
                                  '/' + str(datwep) + \
                                  '_feature_importance_RF.csv')

        res_,res__ = correct_fc_il(df,res, month_to_correct=[6,'CNY', 11], thrsh=0.05)
        resfinal = pd.concat([resfinal, res_]) 
        resfinal = pd.concat([resfinal, res__])  
    return resfinal,feature_importance_df


# In[17]:


def Get_actual(resfinal,df):

    resfinal_brand = resfinal.groupby(['country_brand','date_to_predict','label','horizon'])['prediction'].sum().reset_index()

    df_brand = df.groupby(['country_brand','label','calendar_yearmonth'])['offtake'].sum().reset_index()
    res = pd.merge(left = resfinal_brand, 
                   right = df_brand[['calendar_yearmonth','country_brand','label','offtake']],
                   left_on = ['date_to_predict','country_brand','label'],
                   right_on = ['calendar_yearmonth','country_brand','label'], 
                   how = 'left') 

    res['forecasted_month'] = pd.to_datetime(res['date_to_predict'],format = '%Y%m')  
    res['actual_month'] = res.apply (lambda x: x.forecasted_month-relativedelta(months = x.horizon), axis = 1)  
    res['cycle_month'] = res.apply (lambda x: x.actual_month+relativedelta(months = 2), axis = 1)  

    res['forecasted_month'] = res['forecasted_month'].apply(lambda x: x.strftime("%Y-%m")) 
    res['actual_month'] = res['actual_month'].apply(lambda x: x.strftime("%Y-%m")) 
    res['cycle_month'] = res['cycle_month'].apply(lambda x: x.strftime("%Y-%m"))   

    res['scope'] = res['label']
    df_il = res[res.scope=='il']
    
    return df_il


# In[18]:


def format_res(df_il):
    df_il.rename(columns = {'prediction':'forecast',
                            'offtake':'actual'},inplace = True) 
    df_il["country"] = df_il["country_brand"].apply(lambda x: x.split("_")[0])
    df_il["brand"] = df_il["country_brand"].apply(lambda x: x.split("_")[1])
    df_il.loc[df_il.forecasted_month.isin(['2019-04','2019-05','2019-06']),'actual']=0
    df_il.loc[df_il.forecasted_month.isin(['2019-04','2019-05','2019-06']),'forecast']=0
    return df_il


# In[19]:


def calculate_KPI(df_il,name):
    
    ytd_forecasts_kpis = pd.DataFrame(columns=["kpi_name", "horizon", "scope", "level", 
                                               "computed_on", "kpi_type", "month", "value"])

    year = '2019'
    months_2019 = ['%s-%.2i' %(year, month) for month in range(1,13,1)]

    year = '2018'
    months_2018 = ['%s-%.2i' %(year, month) for month in range(1,13,1)]

    horizons = [1, 4, 'R6M']
    first_month = None
    verbose = False  
    
    scopes = df_il.scope.unique()

    """
    computed_on: define on which scope to compute the KPIs:
        * 'all_skus': compute the KPIs for all SKUs
        * 'country': compute one KPI per country (name of country given in column "computed_on")
        * 'brand': compute one KPI per brand (name of brand given in column "computed_on"). Note: one brand can be common to several countries
        * 'country_brand': compute one KPI per brand per country (name of country & brand given in column "computed_on")
    """
    computed_on_list = ['all_skus',
                        'country',
                        'brand',
                        'country_brand']

    for computed_on in computed_on_list:
        if computed_on == 'all_skus':
            ytd_forecasts_kpis = fill_kpi_df(ytd_forecasts_kpis, 
                                             df_il, 
                                             computed_on, 
                                             computed_on, 
                                             scopes,
                                             months_2019, 
                                             horizons, 
                                             first_month=first_month, 
                                             verbose=verbose)
        else:
            list_items = df_il[computed_on].unique()
            for item in list_items: 
                df_ = df_il.copy()
                df_ = df_.query("%s=='%s'" %(computed_on, item))
                ytd_forecasts_kpis = fill_kpi_df(ytd_forecasts_kpis, 
                                                 df_, 
                                                 computed_on, 
                                                 item, 
                                                 scopes, 
                                                 months_2019, 
                                                 horizons, 
                                                 first_month=first_month, 
                                                 verbose=verbose) 

    ytd_forecasts_kpis_2019 = ytd_forecasts_kpis[(ytd_forecasts_kpis.month>='2019-01')&(ytd_forecasts_kpis.month<='2019-12')] 

    df_kpi = ytd_forecasts_kpis_2019.copy() 
    df_kpi.to_csv('../blending/2019/kpi_panel_upgrade_0610_{0}.csv'.format(name),index = False)
    df_kpi['version'] = name
    kpis_new = kpi_stats.KPI_formalization(df_kpi,kpi_name = 'bias',skip_month = '2019-06', version = name)   
    kpis_new.generate_KPI()


# In[20]:


def output_res_kpi(df_il,config):
    
    df_il = format_res(df_il)
    df_il.to_csv('../blending/2019/res_use_blending_0610_{0}.csv'.format('RF'),index = False)  
    calculate_KPI(df_il,'RF')

def output_feature_importance(config,feature_importance_df):
    feature_importance_df_sets = dict() 
    for h in range(1,13):
        feature_importance_df_sets[h] = feature_importance_df[feature_importance_df.horizon==h].\
        sort_values(['importance'], ascending = False) [:50]['index'].values
    feature_importance_df_sets_ = pd.DataFrame(feature_importance_df_sets)
    if config["FirstRun"]:
        feature_importance_df_sets_.to_csv(config['temp_folder_path'] + config["feature_import_first_run"])


# In[21]:


if __name__ == '__main__':

    # 1.Load data
    df = Load_raw_master(config)                                        ## 1.1 Load raw_master
    df_feature_addon = Load_category(config)                            ## 1.2 Load category and category forecast data
    if not config["FirstRun"]: 
        feature_importance_df_sets = Load_50_feature(config)            ## 1.3 Load rank firt 50 features

    # 2.Feature Engineering
    dwp_test,dwp,dtp = Create_date_list(config)                         ## 2.1 Create train & test date list
    df = Format_label(df)                                               ## 2.2 Format raw master for label
    table_with_features = Create_feature(df, dwp, dtp)                  ## 2.3 Generate Features
    table_with_features = Calculate_horizon(table_with_features)        ## 2.4 Calculate Horizon
    table_all_features = Merge_table_with_category(table_with_features,
                                                   df_feature_addon)    ## 2.5 Merge category features

    # 3.Train Model
    resfinal,feature_importance_df = Generate_Forecast_IL(
        config,dwp_test,table_all_features,df)                         ## 3.1 Train model and output the forecast
    output_feature_importance(config,feature_importance_df)            ## 3.2 Output feature importance

    # 4.Calculate KPI
    df_il = Get_actual(resfinal,df)                                     ## 4.1 Merge with raw master to get acutal offtake
    output_res_kpi(df_il,config)                                        ## 4.2 Calculate KPI

