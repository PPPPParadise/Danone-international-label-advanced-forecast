# coding: utf-8
"""
This module contains the RawMaster base class. The RawMaster base class is the parent class of all RawMaster subclasses such as
RawMasterDC, RawMasterIL.
"""
import abc
import logging
import os
import pickle

import pandas as pd

from cfg.paths import DIR_CACHE
from src.data_wrangling.Data import Data

logger = logging.getLogger()


class RawMaster:
    """
    This contains the readers of raw data, should they come from files on FS or from Impala on SmartData.
    """

    def __init__(self, raw_data: Data):
        self._raw_data = raw_data

        # todo Zhaoxia: find root cause and remove hot fix
        if 'IL_OFFTAKE' in self._raw_data._data_dict:
            idx = self._raw_data['IL_OFFTAKE']['status'] == '7/1/2019'
            self._raw_data['IL_OFFTAKE'].loc[idx, 'produced_date'] = pd.to_datetime('7/1/2019')
            self._raw_data['IL_OFFTAKE'].loc[idx, 'status'] = 'actual'
            # self._raw_data['IL_OFFTAKE'].loc[
            #     (self._raw_data['IL_OFFTAKE']['date'] > '2019-04-01')
            #     & (self._raw_data['IL_OFFTAKE']['status'] == 'actual')
            #     ].sort_values('date')

        self._raw_master = self.make_raw_master()

    @staticmethod
    def from_pickle(filename: str) -> 'RawMaster':
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Loading {__class__} from pickle {fpath}...')
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
            return data

    def to_pickle(self, filename: str) -> None:
        fpath = os.path.join(DIR_CACHE, filename)
        logger.info(f'Saving {__class__} to pickle {fpath}...')
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)


    @property
    def df(self) -> pd.DataFrame:
        """ Data frame containing the raw data

        :return: Data frame containing the raw data
        """
        return self._raw_master

    @df.setter
    def df(self, _) -> Exception:
        """ Prevents this property to be set from outside the class.

        :param _: Unused.
        :return: Raises an exception.
        """
        raise AttributeError('Attribute "df" is read-only. Please create a new object.')

    @abc.abstractmethod
    def make_raw_master(self) -> pd.DataFrame:
        """ Abstract base method that needs to be defined for all child-classes.

        :return: Pandas dataframe containing the raw_master data
        """
        pass

