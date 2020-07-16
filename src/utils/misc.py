"""
This module contains various utility functions
"""
import logging
import subprocess

import pandas as pd

log = logging.getLogger()


def set_date_to_first_of_month(date_series: pd.Series) -> pd.Series:
    """ Set date to the first of current month

    :param date_series: pandas series holding dates that we want to manipulate
    :return: pandas series holding dates with day set to 1
    """
    return date_series.apply(lambda x: x.replace(day=1))


def run_cmd(args_list):
    """ Run linux commands
    """
    # import subprocess
    print('Running system command: {0}'.format(' '.join(args_list)))
    proc = subprocess.Popen(args_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    s_output, s_err = proc.communicate()
    s_return = proc.returncode
    return s_return, s_output, s_err
