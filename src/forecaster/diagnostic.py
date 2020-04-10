import matplotlib.pyplot as plt
import pandas as pd
from src.forecaster import kpis_computation as kpis
from src.forecaster import kpis_computation
from src.forecaster.utilitaires import fa_score, fa_score_prime, bias_score, add_sku_age
from dateutil.relativedelta import relativedelta


class Diagnostic:
    """ Computes KPIs for model output
    """

    def __init__(self, cvr, raw_master, postprocess='sum', di_eib_il_format=True):
        self.postprocess = postprocess  # can be 'sum', 'diff', 'indep'
        self.raw_master = raw_master
        self.cvr = self.ensure_output_format(cvr, di_eib_il_format)
        if di_eib_il_format:
            self.master_il = add_sku_age(raw_master, 'date', ['sku_wo_pkg'],
                                         ['offtake_di', 'offtake_il', 'offtake_eib'],
                                         Thre_abs=12, Thre_rel=0.02)

    def ensure_output_format(self, cvr, di_eib_il_format):
        """ Load crossval data, converting to new format if required
        :param cvr: String
        :param di_eib_il_format: Boolean whether data has new format or not
        :return: DataFrame
        """

        cvr['date_to_predict'] = pd.to_datetime(cvr['date_to_predict'], format='%Y%m')

        # Selecting right columns names
        if not di_eib_il_format:
            sku_name = ['sku']
            date_name = 'date_to_predict'
            offtake_name = ['offtake_dc']
            offtake_dict = {'offtake_dc': 'target_dc'}
            pass
        else:
            sku_name = ['sku_wo_pkg']
            date_name = 'date'
            offtake_name = ['offtake_di', 'offtake_eib', 'offtake_il']
            offtake_dict = {'offtake_di': 'target_di', 'offtake_eib': 'target_eib', 'offtake_il': 'target_il'}
            cvr = self.convert_format(cvr)

        # computing target
        target = self.raw_master.groupby(['date'] + sku_name, as_index=False).sum()[
            offtake_name + sku_name + ['date']]
        target['date'] = pd.to_datetime(target['date'])
        target = target.rename(columns={'date': date_name}).rename(columns=offtake_dict)

        # merging results
        cvr = cvr.merge(target, how='left', on=[date_name] + sku_name, validate='m:1')
        return cvr

    def convert_format(self, old_cvr):
        """ Convert new format from long to large
        :param old_cvr: DataFrame
        :return: DataFrame
        """

        model_postprocess = self.postprocess

        def extract_label(df, label):
            return df.loc[df.label == label, :].reset_index(drop=True).drop(['label'], axis=1)

        # extracting di, eib, il info
        cvr_di = extract_label(old_cvr, 'di')
        cvr_eib = extract_label(old_cvr, 'eib')
        cvr_il = extract_label(old_cvr, 'il')
        index = ['date_to_predict', 'sku_wo_pkg', 'horizon']

        # depending on the way we want to postprocess values, gives the correct output
        if model_postprocess == 'indep':
            pass
        elif model_postprocess == 'diff':
            cvr_di = (cvr_il.set_index(index) - cvr_eib.set_index(index)).reset_index()
            cvr_di['prediction'] = cvr_di['prediction'].clip(lower=0)
        elif model_postprocess == 'sum':
            cvr_il = (cvr_eib.set_index(index) + cvr_di.set_index(index)).reset_index()
        else:
            raise ValueError('Wrong model_postprocess value')

        def change_prediction_and_target_name(df, label):
            return df.rename(columns={'target': 'target_' + label, 'prediction': 'yhat_' + label + '_calib'}).drop(
                ['month_ratio', 'ratio'], axis=1, errors='ignore')

        # changing columns names to switch from long to wide format
        cvr_di = change_prediction_and_target_name(cvr_di, 'di')
        cvr_eib = change_prediction_and_target_name(cvr_eib, 'eib')
        cvr_il = change_prediction_and_target_name(cvr_il, 'il')

        # gathering label data
        cvr = cvr_di.merge(
            cvr_eib, on=['date_to_predict', 'sku_wo_pkg', 'horizon'], how='inner', validate='1:1').merge(
            cvr_il, on=['date_to_predict', 'sku_wo_pkg', 'horizon'], how='inner', validate='1:1'
        )

        # renaming and adding missing columns
        cvr = cvr.rename(columns={'date_to_predict': 'date'})
        cvr['fold'] = -1  # arbitrary value as this column is not used anymore

        return cvr

    def compute_mx_kpis(self, date_start, date_end, prediction_horizon=6, minimum_histo=-100):
        """ Compute KPIs for chosen date range
        :param minimum_histo: Int
        :param date_start: String
        :param date_end: String
        :param prediction_horizon: Int
        :return: None
        """

        cvr = self.cvr.copy()

        age = self.master_il[['date', 'sku_wo_pkg', 'country', 'brand', 'tier', 'stage', 'age_offtake_di',
                              'age_offtake_il', 'age_offtake_eib']][self.master_il.date >= cvr.date.min()]
        age = age.rename(columns={'date': 'date'})
        age['date'] = pd.to_datetime(age['date'], format='%Y%m')

        cvr = cvr.merge(age, on=['date', 'sku_wo_pkg'], how='left', validate='m:1').fillna(-99)

        for scope in ['il', 'eib', 'di']:
            cvr[f'err_{scope}'] = cvr[f'target_{scope}'] - cvr[f'yhat_{scope}_calib']
            cvr[f'err_{scope}'] = cvr.apply(
                lambda x: x[f'err_{scope}'] if x[f'target_{scope}'] > 0 else 0, axis=1)
            cvr[f'abs_err_{scope}'] = cvr[f'err_{scope}'].abs()
            cvr.loc[cvr[f'age_offtake_{scope}'] < minimum_histo, [f'err_{scope}', f'abs_err_{scope}']] = 0
            cvr.loc[cvr[f'age_offtake_{scope}'] < minimum_histo, [f'target_{scope}', f'yhat_{scope}_calib']] = 0

        filter_cond = (cvr.date >= date_start) & (cvr.date <= date_end) & (cvr.horizon == prediction_horizon)
        cvr_filtered = cvr.loc[filter_cond, :]

        cvr_agg = cvr_filtered.loc[filter_cond, :].sum()

        for scope in ['il', 'eib', 'di']:
            target = cvr_agg[f'target_{scope}']
            pred = cvr_agg[f'yhat_{scope}_calib']
            bias = cvr_agg[f'err_{scope}'] / cvr_agg[f'yhat_{scope}_calib']
            fa = 1.0 - cvr_agg[f'abs_err_{scope}'] / cvr_agg[f'yhat_{scope}_calib']
            fa_prime = 1.0 - cvr_agg[f'abs_err_{scope}'] / cvr_agg[f'target_{scope}']

            print("{:>10}:\tTarget={:>10}\tPred={:>10}\tBias={:>10}\tFA={:>10}\tFA'={:>10}".format(
                scope, int(target), int(pred), round(bias, 4), round(fa, 4), round(fa_prime, 4)))
        return cvr

    def compute_mx_kpis_dc(self, date_start, date_end, prediction_horizon=6):
        cvr = self.cvr.copy()
        date_condition = (cvr.date_to_predict >= date_start) & (cvr.date_to_predict <= date_end)
        horizon_condition = cvr.horizon == prediction_horizon
        res = cvr[date_condition & horizon_condition]
        fa = fa_score(res['target_dc'], res['prediction'])
        faprime = fa_score_prime(res['target_dc'], res['prediction'])
        bi = bias_score(res['target_dc'], res['prediction'])
        print('Bias = {:>7}\tFA = {:>7}\tFA\' = {:>7}'.format(round(bi, 4), round(fa, 4), round(faprime, 4)))
        return cvr

    def plot_kpis(self):
        """ Plot KPIs  for DI-EIB-IL on a monthly basis
        """

        date_start = '2018-01-01'
        date_end = '2019-08-01'

        cvr = self.cvr.copy()

        for scope in ['il', 'eib', 'di']:
            cvr[f'err_{scope}'] = cvr[f'target_{scope}'] - cvr[f'yhat_{scope}_calib']
            cvr[f'abs_err_{scope}'] = cvr[f'err_{scope}'].abs()

        filter_cond = (cvr.date >= date_start) & (cvr.date <= date_end) & (cvr.horizon == 4)
        cvr_filtered = cvr.loc[filter_cond, :]
        cvr_agg = cvr_filtered.groupby('date').sum()

        label_dataframes = {}
        for scope in ['il', 'eib', 'di']:
            dict_date = {}
            for index, row in cvr_agg.iterrows():
                target = row[f'target_{scope}']
                pred = row[f'yhat_{scope}_calib']
                bias = row[f'err_{scope}'] / row[f'yhat_{scope}_calib']
                fa = 1.0 - row[f'abs_err_{scope}'] / row[f'yhat_{scope}_calib']
                fa_prime = 1.0 - row[f'abs_err_{scope}'] / row[f'target_{scope}']
                dict_date[index] = [target, pred, bias, fa, fa_prime]
            dict_date = pd.DataFrame(dict_date, index=['target', 'prediction', 'bias', 'fa', 'fa\'']).T
            label_dataframes[scope] = dict_date

        for key, df in label_dataframes.items():
            print(f'*** Plots for {key} ***')
            df[['target', 'prediction']].plot()
            df[['target', 'prediction']].to_csv(f'{key}_plot.csv', index=False)  # todo remove
            plt.title(f'{key} - target / prediction comparison')
            plt.xlabel('date')
            plt.ylabel('value')
            plt.show()
            df[['fa', 'fa\'']].plot()
            df[['fa', 'fa\'']].to_csv(f'{key}_plot_forecast.csv', index=False)  # todo remove
            plt.xlabel('date')
            plt.ylabel('value')
            plt.title(f'{key} - forecast accuracy')
            plt.show()
            df['bias'].plot()
            df['bias'].to_csv(f'{key}_plot_bias.csv')  # todo remove
            plt.xlabel('date')
            plt.ylabel('value')
            plt.title(f'{key} - bias')
            plt.show()

    def plot_kpis_dc(self):
        """ Plot KPIs for DC on a monthly basis
        """

        date_start = '2018-01-01'
        date_end = '2019-08-01'

        cvr = self.cvr.copy()

        filter_cond = (cvr.date_to_predict >= date_start) & (cvr.date_to_predict <= date_end) & (cvr.horizon == 5)
        cvr_filtered = cvr.loc[filter_cond, :]

        acc_dict = {}
        bias_dict = {}
        for date in cvr_filtered.date_to_predict.unique():
            temp = cvr_filtered[cvr_filtered.date_to_predict == date]
            acc = fa_score(temp['target_dc'], temp['prediction'])
            bias = bias_score(temp['target_dc'], temp['prediction'])
            acc_dict[date] = acc
            bias_dict[date] = bias
        acc_dict = pd.DataFrame(acc_dict, index=['accuracy']).T
        bias_dict = pd.DataFrame(bias_dict, index=['bias']).T

        cvr_agg = cvr_filtered.groupby('date_to_predict', as_index=False).sum()
        cvr_agg[['target_dc', 'prediction']].plot()
        cvr_agg.to_csv('dc_plot.csv', index=False)  # todo remove
        plt.xlabel('date')
        plt.ylabel('value')
        plt.title(f'dc - target / prediction comparison')
        plt.show()

        acc_dict['accuracy'].plot()
        acc_dict['accuracy'].to_csv(f'dc_plot_forecast.csv', index=False)  # todo remove

        plt.title(f'dc - forecast accuracy')
        plt.xlabel('date')
        plt.ylabel('value')
        plt.show()

        bias_dict['bias'].plot()
        bias_dict['bias'].to_csv(f'dc_plot_bias.csv')  # todo remove
        plt.title(f'dc - bias')
        plt.xlabel('date')
        plt.ylabel('value')
        plt.show()

    def run_test(self, plot=False, prediction_horizon=6, MINIMUM_HISTO=-100):
        """ Runs mx_kpis for Y2018 and Q1-2019
        """

        self.cvr.to_csv('data/table_il.csv', index=False)
        print('\n*** Year 2018 ***')
        res2018 = self.compute_mx_kpis(
            '2018-01-01', '2018-12-01', prediction_horizon=prediction_horizon,
            minimum_histo=MINIMUM_HISTO)

        print('\n*** H1 2019 (until December) ***')
        res2019 = self.compute_mx_kpis(
            '2019-01-01', '2019-12-01', prediction_horizon=prediction_horizon,
            minimum_histo=MINIMUM_HISTO)

        if plot:
            self.plot_kpis()

        return pd.concat([res2018, res2019])

    def run_test_dc(self, plot=False, horizon=5):
        """
        Runs mx_kpis for Y2018 and Q1-2019
        """

        self.cvr.to_csv('data/table_dc.csv', index=False)
        print('\n*** Year 2018 ***')
        res2018 = self.compute_mx_kpis_dc('2018-01-01', '2018-12-01', horizon)

        print('\n*** H1 2019 (until July) ***')
        res2019 = self.compute_mx_kpis_dc('2019-01-01', '2019-08-01', horizon)

        if plot:
            self.plot_kpis_dc()

        return pd.concat([res2018, res2019])

    # Transform this data to be compliant with the format expected by ELN functions

    def format_bcg_data(self, forecasts_ori, agg_flag=False, agg_col=None, time_lag=None):
        forecasts = forecasts_ori.copy()
        forecasts.rename(columns={'date': 'forecasted_month', 'target': 'actual',
                                  'yhat': 'forecast', 'sku_wo_pkg': 'sku', 'SKU': "sku"}, inplace=True)
        forecasts['forecasted_month'] = pd.to_datetime(forecasts['forecasted_month'])
        forecasts["forecasted_month"] = forecasts["forecasted_month"].apply(lambda x: x.strftime("%Y-%m"))
        if (type(time_lag) == type(None)):
            forecasts['cycle_month'] = forecasts.apply(
                lambda row: kpis_computation.subtract_month(row['forecasted_month'], int(row['prediction_horizon'])),
                axis=1)
        elif (type(time_lag) == type(1)):
            forecasts['cycle_month'] = forecasts.apply(
                lambda row: kpis_computation.subtract_month(row['forecasted_month'],
                                                            int(row['prediction_horizon'] - time_lag)), axis=1)
        # Reorder columsn #
        #     forecasts = forecasts[["cycle_month", "forecasted_month", "sku", "scope", "forecast", "actual"]]
        #
        if (agg_flag == True):
            cols_sel = ["cycle_month", "forecasted_month", 'country', "sku", "scope", "forecast", "actual"]
            #         cols_add = set(agg_col)-set(cols_sel)
            #         cols_sel.append(list(cols_add))
            cols_sel = list(set(cols_sel + agg_col))
            forecasts = forecasts[cols_sel]
            agg_col.append('scope')
            forecasts_ = forecasts.groupby(agg_col).agg('sum')
            forecasts_ = forecasts_.reset_index()
            forecasts = forecasts_.copy()

        else:
            forecasts = forecasts[["cycle_month", "forecasted_month", "sku", "scope", "forecast", "actual"]]

        # Fill remaining NaN with 0 #
        forecasts = forecasts.fillna(0)

        return forecasts

    def ytd_r6M(self, merged_and_calibrated_btrs, market='IL', first_month='2019-01', month='2019-08', absvalue=True):
        merged_and_calibrated_btrs['date'] = pd.to_datetime(merged_and_calibrated_btrs['date'])
        merged_and_calibrated_btrs.rename(columns={'horizon': 'prediction_horizon'}, inplace=True)
        merged_and_calibrated_btrs['date_m'] = merged_and_calibrated_btrs.apply(
            lambda x: x.date - relativedelta(months=x.prediction_horizon), axis=1)
        merged_and_calibrated_btrs['prediction_horizon'] = merged_and_calibrated_btrs['prediction_horizon'] - 2

        df_di_ = merged_and_calibrated_btrs.copy()
        df_di_.rename(columns={'target_'+str.lower(market): 'target',
                               'yhat_'+str.lower(market)+'_calib': 'yhat'}, inplace=True)
        df_di_['scope'] = market
        df_di_ = self.format_bcg_data(df_di_)

        if absvalue:
            r_bias = kpis.YTD_rolling_bias(6, 3, month, df_di_, scope=market, agg_level='sku',
                                           verbose=True, first_month=first_month)
        else:
            r_bias = kpis.YTD_rolling_bias_woabs(6, 3, month, df_di_, scope=market, agg_level='sku',
                                           verbose=True, first_month=first_month)
        return r_bias
