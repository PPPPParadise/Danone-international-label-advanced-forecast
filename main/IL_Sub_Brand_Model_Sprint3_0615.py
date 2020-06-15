#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re, os, gc
import matplotlib.pyplot as plt
import lunarcalendar 
import src.kpi.KPI_formalization as kpi_stats
import src.forecaster.utilitaires as util
import src.forecaster.modelil as mod 
import src.kpi.kpis_computation as kpis
from typing import List
from functools import reduce
from collections import namedtuple
from scipy import stats
from os import listdir 
from os.path import isfile, join
from dateutil.relativedelta import relativedelta
from src.forecaster.features import *
from collections import namedtuple
from src.forecaster.MLDCModel import MLDCModel
from src.forecaster.features import *
from src.forecaster.model import Model
import warnings  
ModelConfig = namedtuple("ModelConfig", "model_name model_params")
warnings.filterwarnings('ignore')


# #### Load Data

# In[2]:


config = {}

#path 
config["project_folder_path"] = ".."
config["data_folder_path"] = "data"
config["temp_folder_path"] = "temp"
config["result_folder_path"] = "sub_brand"

#input files
config["input_raw_master"] = "raw_master_il_20200210_Update_0602_subbrand_catupdate_v2.csv"
config["input_category_forecast"] = "IL_feature_table_all_0610_cat_fsct.csv"

#temp files
config["feature_import_first_run"] = "feature_importance_df_sets_0615_RF.csv"
config["version_name"] = '0615_with_Q2_40_feature'

# Parameter configuration
config["train_start"] = 201601
config["train_end"] = 201912
config["backtest_start"] = 201801
config["backtest_end"] = 201912
config["FirstRun"] = False 
config['horizon'] = 12

# Columns configuration
config["features_int"] = [
    "date_when_predicting", 
    "label",
    "date_to_predict",
    "target",
    "country", 
    "brand",
    "horizon",
    "country_brand_channel",
    "country_brand",
    "sub_brand",
    "tier"
]

config["features_cat_col"] = [
    '0to6_month_population_mean_3M',
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
    '12to36_month_population_mean_12M'
]   

config["features_cat_fsct_col"] = [
    'upre_fsct',
     'spre_fsct',
    'mainstream_fsct',
    'upre_mean_3M_fsct',
    'spre_mean_3M_fsct',
    'mainstream_mean_3M_fsct'
] 

config["feature_test"] = [
    'label', 'country', 'brand', 'tier', 'country_brand', 'sub_brand',
    'date_when_predicting', 'date_to_predict', 'target', 
    'upre_sales3','upre_sales2', 'upre_sales1',
    'spre_sales3', 'spre_sales2','spre_sales1', 
    'mainstream_sales3', 'mainstream_sales2','mainstream_sales1', 
    'month', 'sin_month', 'cos_month', 
    'ANZ_APT','ANZ_KC', 'DE_APT', 'NL_NC', 'UK_APT', 'UK_C&G',
    'il', 'C&G', 'GD','GT', 'PF', 'PN', 'horizon', 
    'sub_brand_offtake_il_mean_3M','sub_brand_sellin_il_mean_3M', 
    'sub_brand_offtake_di_mean_3M','price_tier_cat_mean_3M', 
    'sub_brand_offtake_il_mean_6M','sub_brand_sellin_il_mean_6M',
    'sub_brand_offtake_di_mean_6M','price_tier_cat_mean_6M', 
    'sub_brand_offtake_il_mean_9M','sub_brand_sellin_il_mean_9M', 
    'sub_brand_offtake_di_mean_9M','price_tier_cat_mean_9M', 
    'sub_brand_offtake_il_mean_12M','sub_brand_sellin_il_mean_12M',
    'sub_brand_offtake_di_mean_12M','price_tier_cat_mean_12M', 
    '0to6_month_population_mean_3M','6to12_month_population_mean_3M', 
    '12to36_month_population_mean_3M','0to6_month_population_mean_6M', 
    '6to12_month_population_mean_6M','12to36_month_population_mean_6M', 
    '0to6_month_population_mean_9M', '6to12_month_population_mean_9M', 
    '12to36_month_population_mean_9M','0to6_month_population_mean_12M', 
    '6to12_month_population_mean_12M','12to36_month_population_mean_12M' 
    'upre_fsct', 'spre_fsct','mainstream_fsct', 'upre_mean_3M_fsct', 
    'spre_mean_3M_fsct','mainstream_mean_3M_fsct'
] 


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
            'max_features':40,
            'n_jobs': 12}) 

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


# In[3]:


def Load_50_feature(config):
    
    feature_path = config["project_folder_path"] + '/' +                   config["temp_folder_path"] + '/' +                   config["feature_import_first_run"]
    feature_importance_df_sets = pd.read_csv(feature_path, index_col = 0)
    
    return feature_importance_df_sets


# In[4]:


def Load_raw_master(config):
    
    raw_master_path = config["project_folder_path"] + '/' +                      config["data_folder_path"] + '/' +                      config["input_raw_master"]
    
    df =  pd.read_csv(raw_master_path)

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
    df['sub_brand_retailer_inv'] = df.groupby(['date','sub_brand'])['retailer_inv'].transform(sum)
    df['sub_brand_sellout'] = df.groupby(['date','sub_brand'])['sellout'].transform(sum)
    df['sub_brand_sp_inv'] = df.groupby(['date','sub_brand'])['sp_inv'].transform(sum)
    df['sub_brand_price'] = df.groupby(['date','sub_brand'])['price'].transform(np.mean)
    df['sub_brand_price_krmb_per_ton_il'] = df.groupby(['date','sub_brand'])['price_krmb_per_ton_il'].transform(np.mean)
    df['sub_brand_value_krmb_il'] = df.groupby(['date','sub_brand'])['value_krmb_il'].transform(np.mean)
    df['volume_ton_il'] = df.groupby(['date','sub_brand'])['volume_ton_il'].transform(np.mean)
    df['uprep'] = df.groupby(['date','sub_brand'])['uprep'].transform(np.mean)
    df['upre'] = df.groupby(['date','sub_brand'])['upre'].transform(np.mean)
    df['spre'] = df.groupby(['date','country_brand'])['spre'].transform(np.mean)
    df['mainstream'] = df.groupby(['date','country_brand'])['mainstream'].transform(np.mean)
    df['anp'] = df.groupby(['date','sub_brand'])['anp'].transform(np.mean)
    df['tp'] = df.groupby(['date','sub_brand'])['tp'].transform(np.mean)
    df['rebate'] = df.groupby(['date','country_brand'])['rebate'].transform(np.mean)
    df['tp_rebate'] = df.groupby(['date','country_brand'])['tp_rebate'].transform(np.mean)
    df['sc_anp'] = df.groupby(['date','country_brand'])['sc_anp'].transform(np.mean)
    df['sc_ts'] = df.groupby(['date','country_brand'])['sc_ts'].transform(np.mean)

    features_search_index = [c for c in df.keys() if ('_si' in c)] 
    for col in features_search_index:
        df[col] = df.groupby(['date','sub_brand'])[col].transform(np.mean)

    brands = [c for c in df.columns if 'sub_brand' in c]  

    sel_col = ['country', 'brand', 'date','tier',
               'isa', 'osa', 
               'total_vol',
               'if_vol', 'fo_vol', 'gum_vol', 'cl_vol', 'il_vol',
               '0to6_month_population', '6to12_month_population',
               '12to36_month_population', 'uprep', 'upre', 'spre',
               'mainstream', 'country_brand','sub_brand',
               'calendar_yearmonth', 'sub_brand_offtake_il', 'sub_brand_offtake_di',
               'sub_brand_offtake_eib', 'sub_brand_sellin_eib', 'sub_brand_retailer_inv',
               'sub_brand_sellout', 'sub_brand_sp_inv', 'sub_brand_sellin_di', 'sub_brand_price',
               'sub_brand_price_krmb_per_ton_il', 'sub_brand_value_krmb_il','sub_brand_sellin_il',
                'anp', 'tp', 'rebate', 'tp_rebate', 'sc_anp', 'sc_ts'] +\
                features_search_index  

    df_ = df.drop_duplicates(['date', 'sub_brand'])
    df_ = df_[sel_col]
    price_tier_dict = {'ANZ_APT_PF':'upre',
                       'ANZ_APT_PN':'spre',              
                       'ANZ_KC_GD':'mainstream', 
                       'ANZ_KC_GT':'spre',
                       'DE_APT_PF':'upre', 
                       'DE_APT_PN':'mainstream',
                       'NL_NC_PN':'mainstream', 
                       'UK_APT_PF':'upre', 
                       'UK_APT_PN':'mainstream',
                       'UK_C&G_C&G':'mainstream'} 

    df_['price_tier_cat']= df_.apply(lambda x:x[price_tier_dict[x.sub_brand]], axis = 1)  
    return df_


# #### Define Function 

# In[5]:


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
    
    ob = temp[(temp[date_name] == (year * 100 + month - 1)) & (temp['sub_brand'] == brand)][value_col].sum() +          temp[(temp[date_name] == (year * 100 + month + 1)) & (temp['sub_brand'] == brand)][value_col].sum()
    ac = temp[(temp[date_name] == (year * 100 + month)) & (temp['sub_brand'] == brand)][value_col].sum() 
    return ac / ob


# In[6]:


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

            forecast_filtered.loc[(forecast_filtered.date_to_predict == year * 100 + month)&(forecast_filtered.sub_brand==brand), 'prediction'] *= mf

    return forecast_filtered 


# In[7]:


def correct_fc_il(all_sales, res, month_to_correct=(6,'CNY', 11), thrsh=0.05):
    """ This function is a post-processing step to the forecast, changing the forecast of some month to
    correspond to past observed ratio
    :param res: the data frame containing forecasts
    :param month_to_correct: the months of forecast where we want to apply a post-process
    :param thrsh: a threshold under which we do not perform any post-processing
    :return:
    """

    sales = all_sales[all_sales.label=='il'].groupby(['calendar_yearmonth', 'label','country','brand','country_brand','tier','sub_brand'])['offtake'].sum().reset_index()
    temp = res.copy()

    years = list((res.date_to_predict // 100).unique())
    tempil = temp[temp.label == 'il']
    for y in years: 
        for m in month_to_correct:
            for a in tempil.sub_brand.unique():
                tempil = apply_forecast_correction(sales=sales, forecast=tempil, forecast_filtered=tempil, label='il',
                                                   brand=a,year=y, month=m)
    return tempil  


# In[8]:


def correct_fc(all_sales,res, month_to_correct=(7, 'CNY', 11), thrsh=0.05):
    """ This function is a post-processing step to the forecast, changing the forecast of some month to
    correspond to past observed ratio
    :param res: the data frame containing forecasts
    :param month_to_correct: the months of forecast where we want to apply a post-process
    :param thrsh: a threshold under which we do not perform any post-processing
    :return:
    """

    sales = all_sales.groupby(['calendar_yearmonth'])['offtake'].sum().reset_index()
    temp = res.copy()
    years = list((res.date_to_predict.astype(int) // 100).unique())
    tempdc = temp.copy()

    for y in years:
        for m in month_to_correct:
            # Post-process for dc
            tempdc = apply_forecast_correction(sales=sales, forecast=temp, forecast_filtered=tempdc, label='dc',
                                               year=y, month=m, thrsh=thrsh)

    return tempdc


# In[9]:


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


# In[10]:


def preformat_table(sales_danone):
    """ Preformat data table to feed into model
    This function allows to precompute useful tables that will be used by the model
    """
#         sales_danone = self.sales_danone.copy()
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
        ['calendar_yearmonth', 'country_brand', 'country', 'brand','tier','sub_brand','label', 'offtake', 'sellin', 'retailer_inv','sp_inv','sellout','price','upre','spre','mainstream','APT_si', 'C&G_si', 'Feihe_si', 'Friso_si', 'KC_si', 'NC_si', 'Weyth_si','brand_si',
        'anp', 'tp', 'rebate', 'tp_rebate', 'sc_anp','sc_ts']]
    return all_sales 


# In[11]:


def _add_rolled_values(df: pd.DataFrame, date_col: str, granularity: List[str], value_cols: List[str], window_sizes: List[int]):
    """ Calculate rolled features (mean, min, max) of selected columns over past k months.

    :param df: Pandas dataframe containing
    :param date_col: Name of date column
    :param granularity: List of granularity columns
    :param value_cols: List of columns for which we want to calculate rolled features
    :param window_sizes: Size of rolling window
    :return: Pandas dataframe containing rolled features
    """

    df = df.sort_values(date_col)

    rolled_dfs = []
    for window_size in window_sizes:
        roll_df = df.set_index(date_col).groupby(granularity, as_index=True)[value_cols].rolling(window_size, min_periods=1)
        roll_df_mean = roll_df.mean().fillna(0).reset_index()
        roll_df_min = roll_df.min().fillna(0).reset_index()
        roll_df_max = roll_df.max().fillna(0).reset_index()

        roll_df_mean.columns = [str(col) + '_mean_%dM' % window_size if col not in [date_col] + granularity else col for col in roll_df_mean.columns]
        roll_df_min.columns = [str(col) + '_min_%dM' % window_size if col not in [date_col] + granularity else col for col in roll_df_min.columns]
        roll_df_max.columns = [str(col) + '_max_%dM' % window_size if col not in [date_col] + granularity else col for col in roll_df_max.columns]

        res_df = pd.concat([roll_df_mean.set_index([date_col] + granularity),
                                roll_df_min.set_index([date_col] + granularity),
                                roll_df_max.set_index([date_col] + granularity)], axis=1)

        rolled_dfs += [res_df]

    concatenated_rolled_dfs = pd.concat(rolled_dfs, axis=1).reset_index()
    df = pd.merge(left=df,
                      right=concatenated_rolled_dfs,
                      on=['date'] + granularity,
                      how='left',
                      validate='one_to_one',
                      suffixes=(False, False))

    return df


# In[12]:


def _add_lagged_features(df: pd.DataFrame, groupby_cols: List[str], lags: List[int], cols_to_lag: List[str]) -> pd.DataFrame:
    """ Add time-lagged values of selected columns

    :param df: Pandas dataframe to which we want to add lagged values.
    :param groupby_cols: Columns by which we want to group data before lagging
    :param lags: List of temporal lags that we want to create values for
    :param cols_to_lag: Columns which we want to lag
    :return: Pandas dataframe with added lagged values.
    """
    for col in cols_to_lag:
        for lag in lags:
            df[col + f'_lag{lag}'] = df.groupby(groupby_cols)[col].apply(pd.Series.shift, periods=lag).bfill()
    return df


# #### Train and BackTesting the Model at Brand level 

# In[13]:


def Format_label(df):
    df_table = df.copy()
    df_table.rename(columns = {'sub_brand_offtake_il':'offtake_il',
                               'sub_brand_offtake_di':'offtake_di',
                               'sub_brand_offtake_eib':'offtake_eib',
                               'sub_brand_sellin_eib':'sellin_eib',
                               'sub_brand_retailer_inv':'retailer_inv',
                               'sub_brand_sellout':'sellout', 
                               'sub_brand_sp_inv':'sp_inv',
                               'sub_brand_sellin_di':'sellin_di',
                               'sub_brand_sellin_il':'sellin_il',
                               'sub_brand_price':'price'}, inplace = True)
    df_table = preformat_table(df_table)  
    return df_table


# In[14]:


def Create_date_list(config):
    
    dwps = util.create_list_period(config["train_start"], config["train_end"], False)
    dwp_test = util.create_list_period(config["backtest_start"], config["backtest_end"], False)
    dwp, dtp = util.get_all_combination_date(dwps, 12) 
    
    return dwp_test,dwp,dtp


# In[15]:


def Create_feature(df, dwp, dtp):
    
    df = df[df.label=='il']

    grouped_offtake2 =  df.groupby(['calendar_yearmonth','label','country','brand', 'tier','country_brand','sub_brand'])[
                    'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
    df1 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_sub_brand_offtake_') 
    target = features_target(grouped_offtake2, dwp, dtp)
#     grouped_offtake2 = df.groupby(['calendar_yearmonth', 'label','country','brand'])[
#                     'offtake'].sum().unstack('calendar_yearmonth').fillna(0) 
#     df2 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_offtake_') 
#     grouped_offtake2 = df.groupby(['calendar_yearmonth', 'label','brand','tier'])[
#                     'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
#     df3 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_brand_tier_offtake_') 
#     grouped_offtake2 = df.groupby(['calendar_yearmonth', 'country','brand','tier'])[
#                     'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
    # df4 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_tier_offtake_') 
    # grouped_offtake2 = df.groupby(['calendar_yearmonth', 'country','brand'])[
    #                 'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
    # df5 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_offtake_') 
    # grouped_offtake2 = df.groupby(['calendar_yearmonth', 'country'])[
    #                 'offtake'].sum().unstack('calendar_yearmonth').fillna(0)
    # df6 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_offtake_')  

    grouped_offtake2 = df.groupby(['calendar_yearmonth','label','country','brand', 'tier','country_brand','sub_brand'])[
                    'sellin'].sum().unstack('calendar_yearmonth').fillna(0)
    df7 = features_amount_sales(grouped_offtake2, dwp, dtp, 'label_country_brand_tier_sellin_') 

    # grouped_offtake2 = df.groupby(['calendar_yearmonth','country_brand'])[
    #                 'sellin'].mean().unstack('calendar_yearmonth').fillna(0)
    # df8 = features_amount_sales(grouped_offtake2, dwp, dtp, 'country_brand_sellin_', timelag=3) 

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
    # one_hot_label = pd.get_dummies(dffinal['label'])
    dffinal = pd.concat([dffinal, 
                         one_hot_brand,
                         one_hot_tier], axis=1)
    
    return dffinal


# In[16]:


def get_rolling(df):

    DEMAND_MODEL_GRANULARITY_COLS = ['sub_brand']
    df_table_feature_rolling  = _add_rolled_values(df= df, 
        date_col='date',
        granularity=DEMAND_MODEL_GRANULARITY_COLS,
        value_cols=['sub_brand_offtake_il', 'sub_brand_sellin_il','sub_brand_offtake_di','price_tier_cat'], 
        window_sizes=[3, 6, 9, 12])

    df_table_feature_rolling ['date_when_predicting'] = pd.to_datetime(df_table_feature_rolling ['date']).dt.year.astype(
                str) + pd.to_datetime(df_table_feature_rolling ['date']).dt.month.astype(str).str.zfill(2)
    df_table_feature_rolling ['date_when_predicting'] = df_table_feature_rolling ['date_when_predicting'].astype(int)

    roll_features_sel = [c for c in df_table_feature_rolling.keys() if 'mean' in c] 
    
    return df_table_feature_rolling,roll_features_sel


# In[18]:


def Load_category(config):
    
    category_path = config["project_folder_path"] + '/' +                    config["data_folder_path"] + '/' +                    config["input_category_forecast"]
    df_feature_addon = pd.read_csv(category_path)
    
    return df_feature_addon


# In[19]:


def Calculate_horizon(table_all_features):
    
    table_all_features['horizon'] = (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.year -
         pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.year) * 12 + \
        (pd.to_datetime(table_all_features.date_to_predict, format='%Y%m').dt.month -
         pd.to_datetime(table_all_features.date_when_predicting, format='%Y%m').dt.month) 
    
    return table_all_features


# In[20]:


def Merge_table_with_category(table_all_features,df_feature_addon):
    
    table_all_features = table_all_features.merge(
        df_feature_addon[config["features_cat_col"] +\
                         config["features_cat_fsct_col"] + 
                         ['label','country_brand','date_when_predicting','date_to_predict']], 
        left_on=['label','country_brand','date_when_predicting','date_to_predict'],
        right_on =['label','country_brand','date_when_predicting','date_to_predict'])
    
    return table_all_features


# In[21]:


def Merge_table_with_rolling(table_all_features,df_table_feature_rolling,roll_features_sel):
    
    table_all_features = table_all_features.merge(
        df_table_feature_rolling[['date_when_predicting','sub_brand']+roll_features_sel], 
        left_on =['date_when_predicting','sub_brand'],
        right_on = ['date_when_predicting','sub_brand'])
    
    return table_all_features


# In[22]:


def model_single_run(config,dwp_test,table_all_features,df):
    
    resfinal = pd.DataFrame()
    feature_importance_df_final = pd.DataFrame()

    # Filter features to train the model
    features = [x for x in table_all_features.keys() if (x not in config["features_int"]) & 
                                                        (x in config["feature_test"])]
    
    for datwep in dwp_test:
        print(datwep) 

        res = pd.DataFrame()
        feature_importance_df = pd.DataFrame()
        for h in range(1, config["horizon"] + 1): 
            print("training model at horizon: " + str(h))

            subdata = table_all_features[(table_all_features.horizon == h) & (~table_all_features.target.isnull())
                                                 & (table_all_features.date_to_predict <= datwep)]
            if not config["FirstRun"]:
                features = list(feature_importance_df_sets[str(h)])+                            config["features_cat_col"] +                            config["features_cat_fsct_col"]

            x_train = subdata[features].values
            y_train = subdata.target 
            print(x_train.shape) 

            data_test = table_all_features[(table_all_features.date_when_predicting == datwep) &
                                                   (table_all_features.horizon == h)].copy()

            x_test = data_test[features].values

            model = MLDCModel(
                        model_name = config["model_config_RandomForestRegressor"].model_name,
                        model_params = config["model_config_RandomForestRegressor"].model_params)

            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            preds = preds.clip(min=0)
            data_test['horizon'] = h
            data_test['prediction'] = preds 
            res = pd.concat([res, 
                             data_test[["label",
                                        "date_to_predict", 
                                        "country",
                                        'brand',
                                        "tier",
                                        "country_brand",
                                        "sub_brand",
                                        "prediction", 
                                        "horizon"]]]) 
            feature_importance = dict(
                        zip(features, zip(model.feature_importances_, [h] * len(model.feature_importances_))))
            feature_importance = pd.DataFrame(feature_importance, index=['importance', 'horizon']).T
            feature_importance_df = feature_importance_df.append(feature_importance.reset_index(),
                                                                         ignore_index=True)

        feature_importance_df['date_when_preidct'] = datwep
        feature_importance = feature_importance_df
        res_ = correct_fc_il(df_,res, month_to_correct=[6, 'CNY', 11], thrsh=0.05)
        resfinal = pd.concat([resfinal, res_])   
    return resfinal,feature_importance_df 


# In[23]:


def output_feature_importance(config,feature_importance_df):
    feature_importance_df_sets = dict() 
    for h in range(1,13):
        feature_importance_df_sets[h] = feature_importance_df[feature_importance_df.horizon==h].        sort_values(['importance'], ascending = False) [:50]['index'].values
    feature_importance_df_sets_ = pd.DataFrame(feature_importance_df_sets)
    if config["FirstRun"]:
        feature_importance_df_sets_.to_csv(config["project_folder_path"] +                                           '/' + config['temp_folder_path'] +                                           '/' + 'feature_importance_df_sets_0615_RF.csv')


# In[24]:


def format_result_cal_kpi_score(resfinal,Test_name,df,config):
    resfinal_brand = resfinal.groupby(['sub_brand','date_to_predict','label','horizon'])['prediction'].sum().reset_index()
    df_brand = df.groupby(['sub_brand','label','calendar_yearmonth'])['offtake'].sum().reset_index()
    res = pd.merge(left = resfinal_brand, right = df_brand[['calendar_yearmonth','sub_brand','label','offtake']],
                   left_on = ['date_to_predict','sub_brand','label'],
                   right_on = ['calendar_yearmonth','sub_brand','label'], 
                   how = 'left')   
    res['forecasted_month'] = pd.to_datetime(res['date_to_predict'],format = '%Y%m')  
    res['actual_month'] = res.apply (lambda x: x.forecasted_month-relativedelta(months = x.horizon), axis = 1)  
    res['cycle_month'] = res.apply (lambda x: x.actual_month+relativedelta(months = 2), axis = 1)  

    res['forecasted_month'] = res['forecasted_month'].apply(lambda x: x.strftime("%Y-%m")) 
    res['actual_month'] = res['actual_month'].apply(lambda x: x.strftime("%Y-%m")) 
    res['cycle_month'] = res['cycle_month'].apply(lambda x: x.strftime("%Y-%m"))  
    ytd_forecasts_kpis = pd.DataFrame(columns=["kpi_name", "horizon", "scope", "level", "computed_on", "kpi_type", "month", "value"])
    year = '2019'
    months_2019 = ['%s-%.2i' %(year, month) for month in range(1,13,1)]

    year = '2018'
    months_2018 = ['%s-%.2i' %(year, month) for month in range(1,13,1)]

    horizons = [1, 4, 'R6M']
    first_month = None
    verbose = False  
    df_il = res.copy() 
    df_il['scope'] =df_il['label']
    df_il = df_il[df_il.scope=='il'] 
    df_il.rename(columns = {'prediction':'forecast',
                           'offtake':'actual'},inplace = True) 
    df_il["country_brand"] = df_il["sub_brand"].apply(lambda x: x.split("_")[0]+'_'+x.split("_")[1])
    df_il["country"] = df_il["country_brand"].apply(lambda x: x.split("_")[0])
    df_il["brand"] = df_il["country_brand"].apply(lambda x: x.split("_")[1])  
    df_il.to_csv(config["project_folder_path"] + '/' +                 config["result_folder_path"] + '/' +                 'res' + Test_name + '.csv',index = False) 
#     df_il.loc[df_il.forecasted_month.isin(['2019-04','2019-05','2019-06']),'actual']=0
#     df_il.loc[df_il.forecasted_month.isin(['2019-04','2019-05','2019-06']),'forecast']=0 
    
    # Compute YTD KPIs for different months
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
                        'country_brand',
                        'sub_brand']

    for computed_on in computed_on_list:
        if computed_on == 'all_skus':
            ytd_forecasts_kpis = fill_kpi_df(ytd_forecasts_kpis, df_il, computed_on, computed_on, scopes, months_2019, horizons, first_month=first_month, verbose=verbose)

        else:
            list_items = df_il[computed_on].unique()
            for item in list_items: 
                df_ = df_il.copy()
                df_ = df_.query("%s=='%s'" %(computed_on, item))
                ytd_forecasts_kpis = fill_kpi_df(ytd_forecasts_kpis, df_, computed_on, item, scopes, months_2019, horizons, first_month=first_month, verbose=verbose)   
    ytd_forecasts_kpis_2019 = ytd_forecasts_kpis[(ytd_forecasts_kpis.month>='2019-01')&(ytd_forecasts_kpis.month<='2019-12')]
    ytd_forecasts_kpis_2019.to_csv(config["project_folder_path"] + '/' +                                   config["result_folder_path"] + '/' +                                   'kpi_' +  Test_name + '.csv',index = False)         
    df_kpi = ytd_forecasts_kpis_2019.copy() 
    df_kpi['version'] = Test_name
    kpis_without_Q2 = kpi_stats.KPI_formalization(df_kpi,kpi_name = 'bias',skip_month = '2019-06', version = Test_name) 
    kpis_without_Q2.generate_KPI() 


# In[ ]:


if __name__ == '__main__':
    
    # 1.Load data
    df = Load_raw_master(config)                                          ## 1.1 Load raw_master
    df_feature_addon = Load_category(config)                              ## 1.2 Load category and category forecast data
    if not config["FirstRun"]: 
        feature_importance_df_sets = Load_50_feature(config)              ## 1.3 Load rank firt 50 features
    
    # 2.Feature Engineering
    df_table_feature_rolling,roll_features_sel = get_rolling(df)          ## 2.1 Calculate rolling fetures
    dwp_test,dwp,dtp = Create_date_list(config)                           ## 2.2 Calculate date list
    
    df_ = Format_label(df)                                                ## 2.3 Format tables
    table_with_features = Create_feature(df_, dwp, dtp)                   ## 2.4 Create features
    table_with_features = Calculate_horizon(table_with_features)          ## 2.5 Calculate horizons 
    table_all_features = Merge_table_with_rolling(table_with_features,
                                                  df_table_feature_rolling,
                                                  roll_features_sel)       ## 2.6 Merge category features
    table_all_features = Merge_table_with_category(table_all_features,
                                                   df_feature_addon)       ## 2.7 Merge category features
    # 3.Train Model
    resfinal,feature_importance_df = model_single_run(
        config,dwp_test,table_all_features,df)                             ## 3.1 Train model and output the forecast
    output_feature_importance(config,feature_importance_df)                ## 3.2 Output feature importance
    format_result_cal_kpi_score(resfinal,config["version_name"],df_,config)## 3.3 Output KPI score

