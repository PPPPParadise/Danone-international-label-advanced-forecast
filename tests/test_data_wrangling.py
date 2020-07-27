# coding: utf-8
import logging
import os
import sys
import unittest

import yaml

from cfg.paths import DIR_CFG, LOG_FILENAME, DIR_TEST_DATA, LOG_FORMAT
from src.data_wrangling.Data import Data
from src.data_wrangling.RawMasterIL import RawMasterIL

logging.basicConfig(filename=LOG_FILENAME % 'data_wrangling', format=LOG_FORMAT, level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @unittest.skip
    def test_from_files(self):
        with open(os.path.join(DIR_CFG, 'data.yaml'), 'r') as ymlf:
            files_cfg = yaml.load(ymlf)
        data_il = Data.from_files(
            label='IL'
        )
        data_dc = Data.from_files(
            label='DC'
        )
        self.assertEqual(len(data_il._data_dict), 16)
        self.assertEqual(len(data_dc._data_dict), 12)

    # @unittest.skip
    def test_from_impala(self):
        with open(os.path.join(DIR_CFG, 'impala.yml'), 'r') as ymlf:
            cfg = yaml.load(ymlf)
        data = Data.from_impala(label='DC', cfg=cfg)
        # self.assertEqual(len(data._data_dict), 2)
        print(data._data_dict['POS'].transpose())

    @unittest.skip
    def test_to_from_pickle(self):
        with open(os.path.join(DIR_TEST_DATA, 'il_files.yml'), 'r') as ymlf:
            files_cfg = yaml.load(ymlf)
        data = Data.from_files(label='IL')
        filename = 'pickled_test_data.pkl'
        data.to_pickle(filename)
        data_reloaded = Data.from_pickle(filename)
        self.assertEqual(len(data_reloaded._data_dict), 8)

    @unittest.skip
    def test_raw_master(self):
        with open(os.path.join(DIR_CFG, 'data.yaml'), 'r') as ymlf:
            files_cfg = yaml.load(ymlf)
        data = Data.from_files(
            label='IL'
        )
        raw_master = RawMasterIL(raw_data=data)
        self.assertEqual(len(raw_master.df.columns), 35)


if __name__ == '__main__':
    unittest.main()
