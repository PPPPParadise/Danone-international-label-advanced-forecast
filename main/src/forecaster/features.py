from dateutil.relativedelta import relativedelta
from datetime import datetime
import numpy as np
import pandas as pd
from src.forecaster.utilitaires import add_period , substract_period


def features_amount_sales(data, dates_when_predicting, dates_to_predict, aggreg, timelag=12):
    """ Adding the sales column to the prediction
    """

    to_merge = pd.DataFrame()
    for i in range(len(dates_when_predicting)):
        filter_col = [col for col in data if (col <= dates_when_predicting[i])]
        datatemp = data[filter_col]
        datatemp = datatemp[datatemp.columns[-timelag:]]
        datatemp.columns = map((aggreg + 'sales{}').format, range(timelag, 0, -1))
        datatemp['date_when_predicting'] = dates_when_predicting[i]

        datatemp['date_to_predict'] = dates_to_predict[i]
        datatemp.set_index('date_when_predicting', append=True, inplace=True)
        datatemp.set_index('date_to_predict', append=True, inplace=True)
        to_merge = pd.concat([to_merge, datatemp])

    return to_merge.reset_index()


def features_target(data, dates_when_predicting, dates_to_predict):
    """ Adding the groundtruth column to the predictions
    :param data: dataframe containing sales data
    :param dates_when_predicting: the dates at which the predictions needs to be made
    :param dates_to_predict: the dates for which we are trying to predict the amount of shipments
    """

    to_merge = pd.DataFrame()
    for i in range(len(dates_to_predict)):
        if dates_to_predict[i] in data.columns:
            datatemp = data[dates_to_predict[i]].to_frame()
            datatemp.columns = ['target']
            datatemp['date_when_predicting'] = dates_when_predicting[i]
            datatemp.set_index('date_when_predicting', append=True, inplace=True)

            datatemp['date_to_predict'] = dates_to_predict[i]
            datatemp.set_index('date_to_predict', append=True, inplace=True)
            to_merge = pd.concat([to_merge, datatemp])

    return to_merge.reset_index()


def features_target_r6m(data, dates_when_predicting):
    """ Adding the groundtruth column to the predictions on the R6M prediction
        R6M prediction is the prediction of the cumulative sales between t+5 and t+10
        No need for date_to_predicts since the horizon is fixed
        :param data: dataframe containing sales data
        :param dates_when_predicting: the dates at which the predictions needs to be made
        """

    to_merge = pd.DataFrame()

    # Reformat data on a column form

    df = data.copy().reset_index()
    id_cols = ['country', 'brand', 'tier', 'stage', 'label', 'sku_wo_pkg']

    df = pd.melt(
        df,
        id_vars=id_cols,
        value_vars=list(set(df.columns) - set(id_cols)),
        var_name='date',
        value_name='sales'
    )

    for dwp in set(dates_when_predicting):
        start_date = int(
            (datetime.strptime(str(dwp), "%Y%m") + relativedelta(months=5)).strftime(
                "%Y%m"))
        end_date = int(
            (datetime.strptime(str(dwp), "%Y%m") + relativedelta(months=10)).strftime(
                "%Y%m"))

        dtp = end_date

        # The target is the cum sales between dwp + 5 and dwp + 10
        datatemp = (
            df
                .query(f"date >= {start_date} and date <= {end_date}")
                .groupby(id_cols)
                .agg({"sales": sum})
                .rename(columns={"sales": "target_r6m"})
        )

        datatemp['date_when_predicting'] = dwp
        datatemp.set_index('date_when_predicting', append=True, inplace=True)

        datatemp['date_to_predict'] = dtp
        datatemp.set_index('date_to_predict', append=True, inplace=True)

        to_merge = pd.concat([to_merge, datatemp])

    return to_merge.reset_index()


def create_seasonality_features(dates_to_predict):
    """ Adding seasonality feature column to the data set
    """
    dates_to_predict = list(set(dates_to_predict))
    df = pd.DataFrame()
    df['date_to_predict'] = dates_to_predict

    df['month'] = df.date_to_predict.apply(lambda x: (x % 100) / 12)
    df['sin_month'] = np.sin(np.pi * df['month'])
    df['cos_month'] = np.cos(np.pi * df['month'])

    return df[['date_to_predict', 'month', 'sin_month', 'cos_month']]


def create_cat_features(data, dates_to_predict):
    """ Adding the groundtruth column to the predictions
    :param data: dataframe containing sales data
    :param dates_to_predict: the dates for which we are trying to predict the amount of shipments
    """

    data2 = data.copy()
    data2['calendar_yearmonth'] = pd.to_datetime(data2['date']).dt.year.astype(
        str) + pd.to_datetime(
        data2['date']).dt.month.astype(str).str.zfill(2)
    data2['calendar_yearmonth'] = data2['calendar_yearmonth'].astype(int)
    to_merge = pd.DataFrame()
    for i in range(len(dates_to_predict)):
        datatemp = data2[data2.calendar_yearmonth == dates_to_predict[i]][
            ['total_vol', 'if_vol', 'fo_vol', 'gum_vol', 'cl_vol', 'il_vol',
             '0to6_month_population', '6to12_month_population', '12to36_month_population']].drop_duplicates().copy()

        datatemp['date_to_predict'] = dates_to_predict[i]
        to_merge = pd.concat([to_merge, datatemp])

    return to_merge


def features_sell_in_fc(sell_in_fc, granularity, dates_when_predicting, dates_to_predict, delta_window=3):
    """
    """
    sell_in_fc_period = sell_in_fc[sell_in_fc['cycle_month'].isin(dates_when_predicting)]
    temp = sell_in_fc_period.groupby(['cycle_month'] + granularity)[
        'forecast'].sum().reset_index()

    # temp = temp[temp['calendar_yearmonth'].isin(dates_to_predict)]
    temp.rename(columns={'cycle_month': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearmonth': 'date_to_predict'}, inplace=True)

    res = temp[temp.date_to_predict.isin(dates_to_predict)]
    stackedtemp = temp.copy()
    list_horizons = list(np.arange(-delta_window, 0)) + list(np.arange(1, delta_window + 1))

    for i in list_horizons:
        temp2 = temp.copy()
        if i < 0:
            temp2['date_to_predict'] = temp2['date_to_predict'].apply(lambda x: substract_period(x, np.abs(i), highest_period=12))
        else:
            temp2['date_to_predict'] = temp2['date_to_predict'].apply(lambda x: add_period(x, i, highest_period=12))
        temp2.rename(columns={'forecast': 'forecast_' + str(i)}, inplace=True)
        granmerge = [x for x in granularity if x != 'calendar_yearmonth']
        stackedtemp = pd.merge(stackedtemp, temp2[
            granmerge + ['date_when_predicting', 'date_to_predict', 'forecast_' + str(i)]], how='left',
                               on=granmerge + ['date_when_predicting', 'date_to_predict'])
    final = pd.merge(res, stackedtemp.drop(columns=['forecast']), how='left',
                     on=granmerge+['date_when_predicting', 'date_to_predict'])

    return final


def features_eln_fc(sell_in_fc, granularity, dates_when_predicting, dates_to_predict, delta_window=3):
    """
    """
    sell_in_fc_period = sell_in_fc[sell_in_fc['cycle_month'].isin(dates_when_predicting)]
    temp = sell_in_fc_period.groupby(['cycle_month'] + granularity)[
        'forecast'].sum().reset_index()

    # temp = temp[temp['calendar_yearmonth'].isin(dates_to_predict)]
    temp.rename(columns={'cycle_month': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'forecast': 'forecast_eln'}, inplace=True)
    temp.rename(columns={'calendar_yearmonth': 'date_to_predict'}, inplace=True)

    res = temp[temp.date_to_predict.isin(dates_to_predict)]
    stackedtemp = temp.copy()
    list_horizons = list(np.arange(-delta_window, 0)) + list(np.arange(1, delta_window + 1))

    for i in list_horizons:
        temp2 = temp.copy()
        if i < 0:
            temp2['date_to_predict'] = temp2['date_to_predict'].apply(lambda x: substract_period(x, np.abs(i), highest_period=12))
        else:
            temp2['date_to_predict'] = temp2['date_to_predict'].apply(lambda x: add_period(x, i, highest_period=12))
        temp2.rename(columns={'forecast_eln': 'forecast_eln_' + str(i)}, inplace=True)
        granmerge = [x for x in granularity if x != 'calendar_yearmonth']
        stackedtemp = pd.merge(stackedtemp, temp2[
            granmerge + ['date_when_predicting', 'date_to_predict', 'forecast_eln_' + str(i)]], how='left',
                               on=granmerge + ['date_when_predicting', 'date_to_predict'])
    final = pd.merge(res, stackedtemp.drop(columns=['forecast_eln']), how='left',
                     on=granmerge+['date_when_predicting', 'date_to_predict'])

    return final


def feature_sumopen_orders(hfa, granularity, dates_when_predicting, dates_to_predict):
    """
    Sum of open orders from date when predicting until date to predict
    """
    hfa_period = hfa[hfa['cycle_month'].isin(dates_when_predicting)]
    temp = hfa_period.groupby(['cycle_month'] + granularity)[
        'calendar_yearmonth'].sum().reset_index()

    temp = temp[temp['calendar_yearmonth'].isin(dates_to_predict)]
    temp.rename(columns={'cycle_month': 'date_when_predicting'}, inplace=True)
    temp.rename(columns={'calendar_yearmonth': 'date_to_predict'}, inplace=True)

    res = pd.DataFrame()

    for w in temp['date_when_predicting'].unique():
        orders_known_atweek = temp[temp['date_when_predicting'] == w].copy()
        orders_known_atweek['calendar_yearmonth'] = orders_known_atweek['calendar_yearmonth'].fillna(0)

        sumvalues = orders_known_atweek.groupby(['plant', 'customer_planning_group', 'lead_sku'])[
            'calendar_yearmonth'].cumsum()
        orders_known_atweek['calendar_yearmonth'] = sumvalues.values
        orders_known_atweek.rename(columns={'calendar_yearmonth': 'sum-fc_sellin'}, inplace=True)
        orders_known_atweek['date_when_predicting'] = w
        res = pd.concat([res, orders_known_atweek])

    return res