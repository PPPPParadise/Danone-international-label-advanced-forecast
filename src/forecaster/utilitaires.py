import os
from typing import List

import lunarcalendar
import numpy as np
import pandas as pd

V_VERY_SMALL = 1e-6

path_to_data_folder = os.path.join(os.getcwd(), '.cache')


def create_list_period(start: int, end: int, is_weekly_mode: bool = False) -> List:
    """ This function returns the list of date between start and end example usage : create_list_period(201801, 201852)
    Inputs:
        - start: integer indicating the first date to return
        - end: integer indicating the last date to return
        -is_weekly_mode: boolean that indicates if start and end are indicating weeks or months
    Returns:
        - res: the list of date between start and end
    """

    res = list()
    if is_weekly_mode:  # Weekly mode - not used anymore
        while start <= end:
            res.append(start)
            if start % 100 != 52:
                start += 1
            else:
                start = ((start // 100) + 1) * 100 + 1
        return res
    else:  # Monthly mode -
        while start <= end:
            res.append(start)
            if start % 100 != 12:
                start += 1
            else:
                start = ((start // 100) + 1) * 100 + 1
        return res


def add_period(date: int, add: int, highest_period: int = 12) -> int:
    """ This function returns a week (or month) equal to date + add
    Inputs:
        - date: integer indicating a week number or month number
        - add: integer indicating the number of weeks to add
        - is_weekly_mode: boolean that indicates if the date is in week or months
    Returns:
        - res: the resulting operation (a new date, integer)
    """

    i = 0
    while i < add:
        if date % 100 != highest_period:
            date += 1
        else:
            date = ((date // 100) + 1) * 100 + 1
        i += 1
    return date


def substract_period(date, add, highest_period: int=52):
    i = 0
    while i < add:
        if date % 100 != 1:
            date -= 1
        else:
            date = ((date//100)-1)*100 + highest_period
        i += 1
    return date


def get_all_combination_date(dwp: List, horizon: int):
    """ This function creates dates_when_predicting list and its associated date_to_predict taking into account the given
    horizon

    :param dwp: list of dates when predicting
    :param horizon: horizon of prediction
    :return: lists of dwp and dtp
    """

    list_dwp = list()
    list_dtp = list()
    for w in dwp:
        for h in range(1, horizon + 1):
            list_dwp.append(w)
            list_dtp.append(add_period(w, h))
    return list_dwp, list_dtp


def get_observed_sales(res: pd.DataFrame, label: str, year: int, month: int, date_name: str, value_col: str) -> float:
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
    temp = temp[temp.label == label]
    ob = temp[temp[date_name] == (year * 100 + month - 1)][value_col].sum() + \
         temp[temp[date_name] == (year * 100 + month + 1)][value_col].sum()
    ac = temp[temp[date_name] == (year * 100 + month)][value_col].sum()

    return ac / ob


def extend_forecast(forecast: pd.DataFrame, raw_master: pd.DataFrame,
                    di_eib_il_format: bool, n_year_extended_horizon: int=1) -> pd.DataFrame:
    """ This function is used to extend forecasts. It compares forecast total offtake to historical one to compute a ratio
    and uses this ratio to extrapolate a trend over the following years
    :param forecast: dataframe with the model forecasts
    :param raw_master: raw data
    :param di_eib_il_format: which format to uses to read the data
    :param n_year_extended_horizon: number of years used to extend the forecast
    :return: forecast appended with the extrapolation
    """

    # 1. Selecting correct date format
    raw_master['date'] = pd.to_datetime(raw_master['date'], format='%Y-%m-%d')
    filter_horizon = (forecast['prediction_horizon'] >= 1)
    forecast_future = forecast[filter_horizon]

    # 2. Selecting correct column name
    if di_eib_il_format:
        forecast_column = ['yhat_il_calib', 'yhat_di_calib', 'yhat_eib_calib']
        ratio_column = ['yhat_il_calib']
        raw_columns = ['offtake_il']
        correction_name = ['il']
    else:
        forecast_column = ['yhat']
        ratio_column = ['yhat']
        raw_columns = ['offtake_dc']
        correction_name = ['dc']

    # 3.Selecting correct horizon for historical data
    date_histo_start = forecast_future['date'].min() - pd.DateOffset(years=1)
    date_histo_end = forecast_future['date'].max() - pd.DateOffset(years=1)
    date_condition = (raw_master['date'] >= date_histo_start) & (raw_master['date'] <= date_histo_end)
    raw_master_filtered = raw_master[date_condition]

    # 4. Computing total distribution and % correction
    histo_results = raw_master_filtered[raw_columns].rename(columns=dict(zip(raw_columns, correction_name))).sum()
    forecast_results = forecast_future[ratio_column].rename(
        columns=dict(zip(ratio_column, correction_name))).sum()
    correction = forecast_results.sum() / histo_results.sum()

    # 5.Extending forecasts
    extended_forecast = pd.DataFrame()
    extended_horizon = forecast_future.copy()

    for h in range(1, n_year_extended_horizon + 1):
        extended_horizon[forecast_column] *= correction
        extended_horizon['prediction_horizon'] += 12
        extended_horizon['date'] += pd.DateOffset(years=1)
        extended_forecast = extended_forecast.append(extended_horizon, ignore_index=True)

    # Flagging extended forecasts
    # forecast['extended'] = 0
    # extended_forecast['extended'] = 1

    return pd.concat([forecast, extended_forecast], ignore_index=True)


def fa_score(actual_values, predicted_values):
    """ Function to compute fa_score as defined by Danone
    """
    if not predicted_values.sum():
        return np.nan

    bias = predicted_values - actual_values
    return 1 - (np.abs(bias).sum() / predicted_values.sum())


def fa_score_prime(actual_values, predicted_values):
    """ Function to compute fa_score as it is usually defined
    """
    if not predicted_values.sum():
        return np.nan

    bias = predicted_values - actual_values
    return 1 - (np.abs(bias).sum() / actual_values.sum())


def bias_score(actual_values, predicted_values):
    """ Function to compute bias
    """
    if not predicted_values.sum():
        return np.nan

    bias = predicted_values - actual_values
    return - bias.sum() / predicted_values.sum()


def load_file(file_name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(path_to_data_folder, file_name), sep=',')


def add_sku_age(df: pd.DataFrame, date_col: str, groupby_cols: List[str], value_cols: List, Thre_abs: float,
                Thre_rel: float) -> pd.DataFrame:
    """ Add SKU age for given rules
    """

    df_raw_ = df.copy()
    df_raw_[date_col] = pd.to_datetime(df_raw_[date_col])
    # Loop in each label(di, il, eib)
    for col in value_cols:
        launch_date_col_name = f'launch_date_{col}'
        age_col_name = f'age_{col}'

        df_raw_ = df_raw_.sort_values(groupby_cols + [date_col])
        # Find rolling 3 month min for each SKU
        df_raw_min = \
            df_raw_[[col, date_col] + groupby_cols].set_index(date_col).groupby(groupby_cols, as_index=True)[
                col].rolling(3, min_periods=1).min().reset_index()
        df_raw_min.rename(columns={col: f'{col}_min3'}, inplace=True)
        # Find max in all historical data for each SKU
        df_raw_max = df_raw_[[col, date_col] + groupby_cols].groupby(groupby_cols, as_index=True)[col].apply(
            max).reset_index()
        df_raw_max.rename(columns={col: f'{col}_max'}, inplace=True)
        # Merge rolling 3 month min into raw data
        df_raw_ = pd.merge(left=df_raw_, right=df_raw_min, on=groupby_cols + [date_col], how='left',
                           validate='one_to_one',
                           suffixes=(False, False))
        # Merge historical max in raw data
        df_raw_ = pd.merge(left=df_raw_,
                           right=df_raw_max,
                           on=groupby_cols,
                           how='left',
                           validate='many_to_one',
                           suffixes=(False, False)
                           )
        # Find launch month when rolling 3 min value meet requirement:
        # 1) exceed abs volume threshold, e.g,1 ton
        # 2) exceed relative volume threhold: e.g, 5%*historical_max
        cond = (df_raw_[f'{col}_min3'] >= Thre_abs) & (df_raw_[f'{col}_min3'] > df_raw_[f'{col}_max'] * Thre_rel)
        df_raw_launchdate = df_raw_[cond].groupby(groupby_cols)[date_col].min().reset_index()
        df_raw_launchdate.rename(columns={date_col: launch_date_col_name}, inplace=True)

        # Merge launch month into raw data
        df_raw_ = pd.merge(left=df_raw_,
                           right=df_raw_launchdate,
                           on=groupby_cols,
                           how='left',
                           validate='many_to_one',
                           suffixes=(False, False)
                           )
        # Cal age for each SKU
        df_raw_[launch_date_col_name] = pd.to_datetime(df_raw_[launch_date_col_name])
        df_raw_[age_col_name] = (df_raw_[date_col].dt.year - df_raw_[launch_date_col_name].dt.year) * 12 + (
                df_raw_[date_col].dt.month - df_raw_[launch_date_col_name].dt.month)
        # Drop tmp columns
        df_raw_.drop([launch_date_col_name, f'{col}_min3', f'{col}_max'], axis=1, inplace=True)

    df_raw_.sort_values(groupby_cols + [date_col], inplace=True)
    return df_raw_


def format_label_data(data: pd.DataFrame, label: str):
    """ This function formats data to ensure proper proper model loading

    :param data: raw dataframe
    :param label: label to format the data
    :return: formatted data
    """

    data = data.copy()
    data['label'] = label
    data['offtake'] = data['offtake_' + label]
    data['sellin'] = data['sellin_' + label]
    return data


def apply_forecast_correction(sales, forecast, forecast_filtered, label, year, month, thrsh=0.05):
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

    if month == 'CNY':
        month = get_cny_month(year)
        month_y1 = get_cny_month(year_minus_1)
        month_y2 = get_cny_month(year_minus_2)

    correction_condition = (int(str(year * 100 + month)) != forecast.date_to_predict.max()) & \
                           (int(str(year * 100 + month)) != forecast.date_to_predict.min())

    if correction_condition:
        tar = (get_observed_sales(sales, label, year_minus_2, month_y2, 'calendar_yearmonth', 'offtake') + get_observed_sales(
            sales, label, year_minus_1, month_y1, 'calendar_yearmonth', 'offtake')) / 2
        acoc = get_observed_sales(forecast, label, year, month, 'date_to_predict', 'prediction')
        mf = tar / acoc

        if np.abs(tar - acoc) > thrsh:
            forecast_filtered.loc[forecast_filtered.date_to_predict == year * 100 + month, 'prediction'] *= mf

    return forecast_filtered


def convert_format(old_cvr):
    """ Convert new format from long to large for di_eib_il_format

    :param old_cvr: DataFrame
    :return: DataFrame
    """

    def extract_label(df, label):
        return df.loc[df.label == label, :].reset_index(drop=True).drop(['label'], axis=1)

    # Extracting di, eib, il info
    cvr_di = extract_label(old_cvr, 'di')
    cvr_eib = extract_label(old_cvr, 'eib')
    cvr_il = extract_label(old_cvr, 'il')

    def change_prediction_and_target_name(df, label):
        return df.rename(columns={'target': 'target_' + label, 'prediction': 'yhat_' + label + '_calib'}).drop(
            ['month_ratio', 'ratio'], axis=1, errors='ignore')

    # Changing columns names to switch from long to wide format
    cvr_di = change_prediction_and_target_name(cvr_di, 'di')
    cvr_eib = change_prediction_and_target_name(cvr_eib, 'eib')
    cvr_il = change_prediction_and_target_name(cvr_il, 'il')

    # Gathering label data
    cvr = cvr_di.merge(
        cvr_eib, on=['date_to_predict', 'sku_wo_pkg', 'horizon'], how='inner', validate='1:1').merge(
        cvr_il, on=['date_to_predict', 'sku_wo_pkg', 'horizon'], how='inner', validate='1:1'
    )

    # Renaming and adding missing columns
    cvr = cvr.rename(columns={'date_to_predict': 'date'})

    return cvr


def convert_long_to_wide(cvr, raw_master, di_eib_il_format):
    """ Convert data from long to large format and append target to it

    :param cvr: dataframe containing forecasts
    :param raw_master: raw master containing original data
    :param di_eib_il_format: boolean whether we are predicting direct china or ddi eib il
    :return: dataframe with wide format
    """

    cvr['date_to_predict'] = pd.to_datetime(cvr['date_to_predict'], format='%Y%m')

    # Selecting right columns names
    if not di_eib_il_format:
        sku_name = ['sku']
        date_name = 'date_to_predict'
        offtake_name = ['offtake_dc']
        offtake_dict = {'offtake_dc': 'target_dc'}
    else:
        sku_name = ['sku_wo_pkg']
        date_name = 'date'
        offtake_name = ['offtake_di', 'offtake_eib', 'offtake_il']
        offtake_dict = {'offtake_di': 'target_di', 'offtake_eib': 'target_eib', 'offtake_il': 'target_il'}
        cvr = convert_format(cvr)

    # computing target
    target = raw_master.groupby(['date'] + sku_name, as_index=False).sum()[offtake_name + sku_name + ['date']]
    target['date'] = pd.to_datetime(target['date'], format='%Y-%m-%d')
    target = target.rename(columns={'date': date_name}).rename(columns=offtake_dict)

    # merging results
    cvr = cvr.merge(target, how='left', on=[date_name] + sku_name, validate='m:1')
    return cvr


def correct_invalid_skus(forecast: pd.DataFrame, raw_master: pd.DataFrame) -> pd.DataFrame:
    """ Correct prediction for invalid skus. For current data, that mainly relates to di.
    :param forecast: dataframe containing forecasts
    :param raw_master: raw master containing original data
    :return: dataframe containing corrected forecasts
    """
    # Find the valid skus for each scope. we check the full history with year-to-date data.
    # Only the SKUs with all-zeros will be consindered as invalid.
    di_skus_sum = raw_master.groupby('sku_wo_pkg')['offtake_di'].sum().reset_index()
    di_skus_invalid = set(di_skus_sum[di_skus_sum.offtake_di < V_VERY_SMALL].sku_wo_pkg)
    il_skus_sum = raw_master.groupby('sku_wo_pkg')['offtake_il'].sum().reset_index()
    il_skus_invalid = set(il_skus_sum[il_skus_sum.offtake_il < V_VERY_SMALL].sku_wo_pkg)
    eib_skus_sum = raw_master.groupby('sku_wo_pkg')['offtake_eib'].sum().reset_index()
    eib_skus_invalid = set(eib_skus_sum[eib_skus_sum.offtake_eib == 0].sku_wo_pkg)

    forecast.loc[((forecast.label == 'di') & (forecast.sku_wo_pkg.isin(di_skus_invalid))), 'prediction'] = 0
    forecast.loc[((forecast.label == 'il') & (forecast.sku_wo_pkg.isin(il_skus_invalid))), 'prediction'] = 0
    forecast.loc[((forecast.label == 'eib') & (forecast.sku_wo_pkg.isin(eib_skus_invalid))), 'prediction'] = 0

    return forecast
