import numpy as np
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

""" Utils functions """

def add_month(source_month, n):
    """
    Inputs:
      month: string in format 'YYYY-mm', base month
      n: integer (positive)
    
    Returns:
      res: month - n, in format 'YYYY-mm'
    """
    source_month = datetime.strptime(source_month, '%Y-%m')
    res = source_month + relativedelta(months=n)
    res = res.strftime("%Y-%m")
    return res

def subtract_month(source_month, n):
    """
    Inputs:
      month: string in format 'YYYY-mm', base month
      n: integer (positive)
    
    Returns:
      res: month - n, in format 'YYYY-mm'
    """
    source_month = datetime.strptime(source_month, '%Y-%m')
    res = source_month - relativedelta(months=n)
    res = res.strftime("%Y-%m")
    return res

def get_month_diff(month1, month2):
    """
    Gives the number of months between month1 and month2 (month1 - month2)
    
    Inputs:
        month1: string in format format "YYYY-mm", a month
        month2: string in format format "YYYY-mm", a month
        
    Return:
        month_diff: integer
    """
    month1 = datetime.strptime(month1, '%Y-%m')
    month2 = datetime.strptime(month2, '%Y-%m')
    month_diff = (month1.year - month2.year) * 12 + month1.month - month2.month
    return month_diff


""" KPIs computation """

def bias(k, m, data, scope, agg_level="sku", verbose=False):
    """
    M-k bias at month m, at aggregated level
    
    Inputs:
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
      verbose: bool, defines whether informations printed
    """
    
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("Bias M-%s at month %s: No data for the scope %s." %(k, m, scope))
        return None
    
    # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()

    # Compute the forecast computation date: m - k #
    comp_date = subtract_month(m, k)

    # Keep only the forecasts computed in month comp_date #
    data = data.query("cycle_month=='%s'" %comp_date)
    if len(data) == 0:
        if verbose:
            print("Bias M-%s at month %s: No data for the scope %s and computation month %s." %(k, m, scope, comp_date))
        return None

    # Keep only the forecasts for month m
    data = data.query("forecasted_month=='%s'" %m)
    if len(data) == 0:
        if verbose:
            print("Bias M-%s at month %s: No data for the scope %s, computation month %s and forecasted month %s." %(k, m, scope, comp_date, m))
        return None

    data["error"] = data["actual"] - data["forecast"]

    bias = data["error"].sum()/data["forecast"].sum()
    if verbose:
        print("M-%i bias at month %s: %.4f" %(k, m, bias))
        print("Computation details:")
        print("  Forecast computed in month: %s" %comp_date)
        print("  Scope: %s" %scope)
        print("  Aggregated level (all SKUs)")
        
    return bias


def YTD_bias(k, m, data, scope, agg_level="sku", first_month=None, verbose=False):
    """
    Year to date M-k bias at month m, at aggregated level
    
    Inputs:
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      first_month (optional): the YTD bias will be computed from first_month to m.
                              If first_month is None, it is set as Jan of year of month m
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
    """
    
    # Get the first month to be considered in the YTD #
    if first_month == None:
        first_month = "%s-01" %(m.split("-")[0])
        
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("FA M-%s at month %s: No data for the scope %s." %(k, m, scope))
        return None
    
    # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()
    
    if len(data) == 0:
        if verbose:
            print("YTD Bias M-%s from month %s, at month %s: No data for the scope %s." %(k, first_month, m, scope))
        return None
        
    # Get list of months for which forecast have to be taken in the YTD computation #
    list_forecasted_months = [add_month(first_month, i) for i in range(get_month_diff(m, first_month) + 1)]
    
    errors = []
    fcsts = []
    list_comp_date = []
    for forecasted_month in list_forecasted_months: # For each forecasted month:
        # Compute the forecast computation date: m - k #
        comp_date = subtract_month(forecasted_month, k)
        list_comp_date.append(comp_date)
        
        # Keep only the forecasts for month computation_month, computed in comp_date
        data_2 = data.query("cycle_month=='%s'" %comp_date)
        data_2 = data_2.query("forecasted_month=='%s'" %forecasted_month)
        if len(data_2) == 0:
            if verbose:
                print("YTD Bias M-%s from month %s, at month %s: No data for the scope %s, computation month %s and forecasted month %s." %(k, first_month, m, scope, comp_date, forecasted_month))
            return None
        
        # Compute the sum (over SKUs) of fcst and errors for computation_month fcst, using the fcst of month comp_date
        errors.append(sum(data_2["actual"].values - data_2["forecast"].values))
        fcsts.append(data_2["forecast"].sum())
        
    # Total YTD bias: sum of the errors / sum of the forecasts #
    ytd_bias = sum(errors)/sum(fcsts)
    
    if verbose:
        print("YTD M-%i bias at month %s: %.4f" %(k, m, ytd_bias))
        print("Computation details:")
        print("  Using forecasts for month: %s" %list_forecasted_months)
        print("  YTD forecasts computed between month %s and month %s" %(list_comp_date[0], list_comp_date[-1]))
        print("  Scope: %s" %scope)
        print("  Aggregated level (all SKUs)")
        
    return ytd_bias


def FA(k, m, data, scope, agg_level="sku", verbose=False):
    """
    M-k Forecast Accuracy at month m, at aggregated level
    
    Inputs:
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
      verbose: bool, defines whether informations printed
    """
    
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("FA M-%s at month %s: No data for the scope %s." %(k, m, scope))
        return None

    # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()

    # Compute the forecast computation date: m - k #
    comp_date = subtract_month(m, k)

    # Keep only the forecasts computed in month comp_date #
    data = data.query("cycle_month=='%s'" %comp_date)
    if len(data) == 0:
        if verbose:
            print("FA M-%s at month %s: No data for the scope %s and computation month %s." %(k, m, scope, comp_date))
        return None

    # Keep only the forecasts for month m
    data = data.query("forecasted_month=='%s'" %m)
    if len(data) == 0:
        if verbose:
            print("FA M-%s at month %s: No data for the scope %s, computation month %s and forecasted month %s." %(k, m, scope, comp_date, m))
        return None

    data["abs_error"] = data.apply(lambda row: abs(row["actual"] - row["forecast"]), axis=1)
    FA = max(0, 1 - data["abs_error"].sum()/data["forecast"].sum())

    if verbose:
        print("M-%i FA at month %s: %.4f" %(k, m, FA))
        print("Computation details:")
        print("  Forecast computed in month: %s" %comp_date)
        print("  Scope: %s" %scope)
        print("  Aggregated level (all SKUs)")
        
    return FA


def YTD_FA(k, m, data, scope, agg_level="sku", first_month=None, verbose=False):
    """
    Year to date M-k Forecast Accuracy at month m, at aggregated level
    
    Inputs:
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      first_month (optional): the YTD bias will be computed from first_month to m.
                              If first_month is None, it is set as Jan of year of month m
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
    """
        
    # Get the first month to be considered in the YTD #
    if first_month == None:
        first_month = "%s-01" %(m.split("-")[0])
        
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("YTD FA M-%s from month %s, at month %s: No data for the scope %s." %(k, first_month, m, scope))
        return None

     # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()
        
    # Get list of months for which forecast have to be taken in the YTD computation #
    list_forecasted_months = [add_month(first_month, i) for i in range(get_month_diff(m, first_month) + 1)]
    
    abs_errors = []
    fcsts = []
    list_comp_date = []
    for forecasted_month in list_forecasted_months: # For each forecasted month:
        # Compute the forecast computation date: m - k #
        comp_date = subtract_month(forecasted_month, k)
        list_comp_date.append(comp_date)
        
        # Keep only the forecasts for month computation_month, computed in comp_date
        data_2 = data.query("cycle_month=='%s'" %comp_date)
        data_2 = data_2.query("forecasted_month=='%s'" %forecasted_month)
        if len(data_2) == 0:
            if verbose:
                print("YTD FA M-%s at month %s: No data for the scope %s, computation month %s and forecasted month %s." %(k, first_month, m, scope, comp_date, forecasted_month))
            return None
        
        # Compute the sum (over SKUs) of fcst and errors for computation_month fcst, using the fcst of month comp_date
        data_2["abs_error"] = data_2.apply(lambda row: abs(row["actual"] - row["forecast"]), axis=1)
        abs_errors.append(data_2["abs_error"].sum())
        fcsts.append(data_2["forecast"].sum())
        
    # Total YTD bias: sum of the errors / sum of the forecasts #
    ytd_fa = max(0, 1 - sum(abs_errors)/sum(fcsts))
    
    if verbose:
        print("YTD M-%i FA at month %s: %.4f" %(k, m, ytd_fa))
        print("Computation details:")
        print("  Using forecasts for month: %s" %list_forecasted_months)
        print("  YTD forecasts computed between month %s and month %s" %(list_comp_date[0], list_comp_date[-1]))
        print("  Scope: %s" %scope)
        print("  Aggregated level (all SKUs)")
        
    return ytd_fa


def rolling_bias(r, k, m, data, scope, agg_level="sku", verbose=0):
    """
    M-k bias at month m, at aggregated level
    
    Inputs:
      r: integer, rolling period
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
      verbose: bool, defines whether informations printed
    """
    
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("Rolling %s months M-%s bias: No data for the scope %s." %(r, k, scope))
        return None
    
     # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()
        
        
    # Compute the forecast computation date: m - k #
    comp_date = subtract_month(m, r - 1 + k)
     # Keep only the forecasts computed in month comp_date #
    data = data.query("cycle_month=='%s'" %comp_date)
    if len(data) == 0:
        if verbose:
            print("Rolling %s months M-%s bias: No data for the scope %s and computation month %s." %(r, k, scope, comp_date))
        return None
    
    forecasted_months = []
    errors = []
    forecasts = []
    for i in range(r):
        forecasted_month = subtract_month(m, i)
        forecasted_months.append(forecasted_month)

        # Keep only the forecasts for month forecasted_month
        data_2 = data.query("forecasted_month=='%s'" %forecasted_month)
        if len(data_2) == 0:
            if verbose:
                print("Rolling %s months M-%s bias: No data for the scope %s, computation month %s and forecasted month %s." %(r, k, scope, comp_date, forecasted_month))
            return None

        data_2["error"] = data_2["actual"] - data_2["forecast"]
        errors.append(data_2["error"].sum())
        forecasts.append(data_2["forecast"].sum())

    rolling_bias = sum(errors)/sum(forecasts)
    
    if verbose:
        print("Rolling %s months M-%i FA at month %s: %.4f" %(r, k, m, rolling_bias))
        print("Computation details:")
        print("  Forecast computed in month: %s" %comp_date)
        print("  Using forecasts for months %s" %forecasted_months)
        print("  Scope: %s" %scope)
        print("  Aggregated level (all SKUs)")
        if verbose == 2:
            print("  Sum error: %s" %sum(errors))
            print("  Sum forecast: %s" %sum(forecasts))
        
    return rolling_bias


def YTD_rolling_bias(r, k, m, data, scope, agg_level="sku", first_month=None, verbose=0):
    """
    Year to date M-k bias at month m, at aggregated level
    
    Inputs:
      r: integer, rolling period
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      first_month (optional): the YTD bias will be computed from first_month to m.
                              If first_month is None, it is set as Jan of year of month m
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
      verbose: 0, 1 or 2, defines level informations printed
    """
     
    # Get the first month to be considered in the YTD #
    if first_month == None:
        first_month = "%s-01" %(m.split("-")[0])
        
    # Filter data on the scope #
    data = data.query("scope=='%s'" %scope)
    if len(data) == 0:
        if verbose:
            print("YTD R%s Months M-%s from %s, at month %s: No data for the scope %s." %(r, k, first_month, m, scope))
        return None
   
    # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku": # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month","scope"]
        level_all = ["country","brand","tier","stage"]
        level_idx =level_all.index(agg_level)+1
        for i in range(0, level_idx) :
            agg_col_level.insert(i+2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()
        
    # Get list of months for which forecast have to be taken in the YTD computation (months for which rolling KPI computed) #
    list_forecasted_months = [add_month(first_month, i) for i in range(get_month_diff(m, first_month) + 1)]
    
    rolling_errors = []
    rolling_forecasts = []
    computation_dates = []
    for forecasted_month in list_forecasted_months: # For each forecasted month (for which rolling KPI computed)
        if verbose == 2:
            print("Forecasted month: %s" %forecasted_month)
    
        # Compute the forecast computation date: m - k #
        computation_date = subtract_month(forecasted_month, r - 1 + k)
        computation_dates.append(computation_date)
        if verbose == 2:
            print("Computation date: %s" %computation_date)
        # Keep only the forecasts computed in month comp_date #
        data_2 = data.query("cycle_month=='%s'" %computation_date)
        if len(data_2) == 0:
            if verbose:
                print("YTD R%s Months M-%s from %s, at month %s: No data for the scope %s and computation month %s." %(r, k, first_month, m, scope, computation_date))
            return None
    
        rolling_forecasted_months = []
        errors = []
        forecasts = []
        for i in range(r):
            rolling_forecasted_month = subtract_month(forecasted_month, i)
            rolling_forecasted_months.append(rolling_forecasted_month)
            if verbose == 2:
                print("   Rolling forecasted month: %s" %rolling_forecasted_month)

            # Keep only the forecasts for month: forecasted_month
            data_3 = data_2.query("forecasted_month=='%s'" %rolling_forecasted_month)
            if len(data_3) == 0:
                if verbose:
                    print("YTD R%s Months M-%s from %s, at month %s: No data for the scope %s, computation month %s, forecasted month %s and rolling forecasting month %s." %(r, k, first_month, m, scope, computation_date, forecasted_month, rolling_forecasted_month))
                return None

            data_3["error"] = data_3["actual"] - data_3["forecast"]
            errors.append(data_3["error"].sum()) # error of sum of SKUs (aggregated level), for this forecast
            forecasts.append(data_3["forecast"].sum())
            if verbose == 2:
                print("      Error: %s" %data_3["error"].sum())
                print("      Foreacst: %s" %data_3["forecast"].sum())
            
        rolling_errors.append(abs(sum(errors))) # Absolute value of the whole rolling period
        rolling_forecasts.append(sum(forecasts))
        if verbose == 2:
            print("   Absolute sum of errors: %s" %abs(sum(errors)))
            print("   Forecast: %s" %sum(forecasts))

    if verbose == 2:
        print("Sum errors: %s" %sum(rolling_errors))
        print("Sum of forecast: %s\n*** Results: ***" %sum(rolling_forecasts))
    ytd_rolling_bias = sum(rolling_errors)/sum(rolling_forecasts)
    
    if verbose:
        print("YTD Rolling %s months M-%i bias at month %s: %.4f" %(r, k, m, ytd_rolling_bias))
        print("  YTD from month %s to month %s" %(first_month, m))
        print("  Scope: %s" %scope)

    return ytd_rolling_bias


def YTD_rolling_bias_woabs(r, k, m, data, scope, agg_level="sku", first_month=None, verbose=0):
    """
    Year to date M-k bias at month m, at aggregated level

    Inputs:
      r: integer, rolling period
      k: integer, lag between month m and forecast computation
      m: string 'YYYY-MM', month of the KPI (see KPI definition)
      data: datagrame, the dataframe with all skus to be used for the computation and associated forecasts and actuals
      scope: string, 'DC', 'DI', 'IL' or 'EIB'
      first_month (optional): the YTD bias will be computed from first_month to m.
                              If first_month is None, it is set as Jan of year of month m
      agg_level: Aggregation level, either "sku", "stage", "tier", "brand" or "country"
      verbose: 0, 1 or 2, defines level informations printed
    """

    # Get the first month to be considered in the YTD #
    if first_month == None:
        first_month = "%s-01" % (m.split("-")[0])

    # Filter data on the scope #
    data = data.query("scope=='%s'" % scope)
    if len(data) == 0:
        if verbose:
            print("YTD R%s Months M-%s from %s, at month %s: No data for the scope %s." % (r, k, first_month, m, scope))
        return None

    # Aggregate data according to the aggregation level #
    if agg_level not in ["sku", "stage", "tier", "brand", "country"]:
        if verbose:
            print("Invalid aggregation level: plese use \"sku\", \"stage\", \"tier\", \"brand\" or \"country\"")
        return None
    if agg_level != "sku":  # no need to aggregate for sku level (native granularity)
        agg_col_level = ["cycle_month", "forecasted_month", "scope"]
        level_all = ["country", "brand", "tier", "stage"]
        level_idx = level_all.index(agg_level) + 1
        for i in range(0, level_idx):
            agg_col_level.insert(i + 2, level_all[i])
        data = data[agg_col_level + ["forecast", "actual"]].groupby(agg_col_level).sum().reset_index()

    # Get list of months for which forecast have to be taken in the YTD computation (months for which rolling KPI computed) #
    list_forecasted_months = [add_month(first_month, i) for i in range(get_month_diff(m, first_month) + 1)]

    rolling_errors = []
    rolling_forecasts = []
    computation_dates = []
    for forecasted_month in list_forecasted_months:  # For each forecasted month (for which rolling KPI computed)
        if verbose == 2:
            print("Forecasted month: %s" % forecasted_month)

        # Compute the forecast computation date: m - k #
        computation_date = subtract_month(forecasted_month, r - 1 + k)
        computation_dates.append(computation_date)
        if verbose == 2:
            print("Computation date: %s" % computation_date)
        # Keep only the forecasts computed in month comp_date #
        data_2 = data.query("cycle_month=='%s'" % computation_date)
        if len(data_2) == 0:
            if verbose:
                print("YTD R%s Months M-%s from %s, at month %s: No data for the scope %s and computation month %s." % (
                r, k, first_month, m, scope, computation_date))
            return None

        rolling_forecasted_months = []
        errors = []
        forecasts = []
        for i in range(r):
            rolling_forecasted_month = subtract_month(forecasted_month, i)
            rolling_forecasted_months.append(rolling_forecasted_month)
            if verbose == 2:
                print("   Rolling forecasted month: %s" % rolling_forecasted_month)

            # Keep only the forecasts for month: forecasted_month
            data_3 = data_2.query("forecasted_month=='%s'" % rolling_forecasted_month)
            if len(data_3) == 0:
                if verbose:
                    print(
                        "YTD R%s Months M-%s from %s, at month %s: No data for the scope %s, computation month %s, forecasted month %s and rolling forecasting month %s." % (
                        r, k, first_month, m, scope, computation_date, forecasted_month, rolling_forecasted_month))
                return None

            data_3["error"] = data_3["actual"] - data_3["forecast"]
            errors.append(data_3["error"].sum())  # error of sum of SKUs (aggregated level), for this forecast
            forecasts.append(data_3["forecast"].sum())
            if verbose == 2:
                print("      Error: %s" % data_3["error"].sum())
                print("      Foreacst: %s" % data_3["forecast"].sum())

        rolling_errors.append(sum(errors))  # Value of the whole rolling period
        rolling_forecasts.append(sum(forecasts))
        if verbose == 2:
            print("   Sum of errors: %s" % sum(errors))
            print("   Forecast: %s" % sum(forecasts))

    if verbose == 2:
        print("Sum errors: %s" % sum(rolling_errors))
        print("Sum of forecast: %s\n*** Results: ***" % sum(rolling_forecasts))
    ytd_rolling_bias = sum(rolling_errors) / sum(rolling_forecasts)

    if verbose:
        print("YTD Rolling %s months M-%i bias at month %s: %.4f" % (r, k, m, ytd_rolling_bias))
        print("  YTD from month %s to month %s" % (first_month, m))
        print("  Scope: %s" % scope)

    return ytd_rolling_bias
