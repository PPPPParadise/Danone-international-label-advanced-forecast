# coding: utf-8
import logging
import os
import unittest

import pandas as pd
import yaml

from cfg.paths import CFG_ADDL_GRAN, DIR_CACHE
from cfg.paths import LOG_FILENAME
from src.postprocessing.postprocessing import PostProcessing
from src.utils.logging import setup_loggers

logger = logging.getLogger(__name__)
setup_loggers('test_postprocessing', log_filepath=LOG_FILENAME, level=logging.INFO)


class AddGranTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(CFG_ADDL_GRAN) as f:
            cfg = yaml.load(f)

        cls.df_hist_di = pd.read_csv(cfg['di']['history'], sep=';', low_memory=False)
        cls.df_hist_il = pd.read_csv(cfg['il']['history'], sep=';', low_memory=False)
        cls.df_hist_dc = pd.read_csv(cfg['dc']['offtake_history'], sep=';', low_memory=False)
        cls.df_hist_il_sellin = pd.read_csv(cfg['il']['sellin_history'], sep=';', low_memory=False)
        cls.df_hist_eib = pd.read_csv(cfg['il']['eib_history'], sep=';', low_memory=False)
        cls.df_fcst_il_all = pd.read_csv(cfg['il']['forecast'])
        cls.df_fcst_dc = pd.read_csv(cfg['dc']['forecast'])
        cls.df_prodlist = pd.read_csv(cfg['dc']['productlist'], sep=';')
        cls.df_custlist = pd.read_csv(cfg['dc']['customerlist'], sep=';')
        cls.df_hist_dc_sellout = pd.read_csv(cfg['dc']['sellout_history'], sep=';', low_memory=False)
        cls.df_hist_dc_sellin = pd.read_csv(cfg['dc']['sellin_history'], sep=';', low_memory=False)
        cls.df_mapping = pd.read_csv(cfg['il']['mapping'])

    @unittest.skip('Skipped')
    def test_gran_il(self):
        df_fcst_split_il_all, df_actual_all = PostProcessing.add_granularity_il(
            df_hist_di=self.df_hist_di,
            df_hist_il=self.df_hist_il,
            df_hist_il_sellin=self.df_hist_il_sellin,
            df_hist_eib=self.df_hist_eib,
            df_fcst_il_all=self.df_fcst_il_all,
            df_mapping=self.df_mapping
        )

        PostProcessing.detect_dups(df=df_actual_all)
        PostProcessing.detect_dups(df=df_fcst_split_il_all)

        df_fcst_split_il_all.to_csv(os.path.join(DIR_CACHE, 'forecasts_IL.csv'), index=False)
        df_actual_all.to_csv(os.path.join(DIR_CACHE, 'actuals_IL.csv'), index=False)

        self.assertFalse(len(df_fcst_split_il_all.index) == 0)
        self.assertFalse(len(df_actual_all.index) == 0)
        self.assertFalse(df_fcst_split_il_all.isnull().values.any())
        self.assertFalse(df_actual_all.isnull().values.any())

    @unittest.skip('Skipped')
    def test_gran_dc(self):
        df_fcst_split_dc, df_actual_all = PostProcessing.add_granularity_dc(
            df_hist_dc=self.df_hist_dc,
            df_fcst_dc=self.df_fcst_dc,
            df_hist_sellout=self.df_hist_dc_sellout,
            df_hist_sellin=self.df_hist_dc_sellin,
            df_prodlist=self.df_prodlist,
            df_custlist=self.df_custlist,
        )

        PostProcessing.detect_dups(df=df_actual_all)
        PostProcessing.detect_dups(df=df_fcst_split_dc)

        df_fcst_split_dc.to_csv(os.path.join(DIR_CACHE, 'forecasts_DC.csv'), index=False)
        df_actual_all.to_csv(os.path.join(DIR_CACHE, 'actuals_DC.csv'), index=False)

        self.assertFalse(len(df_fcst_split_dc.index) == 0)
        self.assertFalse(len(df_actual_all.index) == 0)
        self.assertFalse(df_fcst_split_dc.isnull().values.any())
        self.assertFalse(df_actual_all.isnull().values.any())

    def test_detect_dups(self):
        def contains_no_dups():
            df = pd.DataFrame({
                'A': [1, 2, 2],
                'B': [3, 4, 5],
                'v': [8, 9, 9]
            })
            PostProcessing.detect_dups(df=df, group_cols=['A', 'B'])

        contains_no_dups()

        def contains_dups():
            df = pd.DataFrame({
                'A': [1, 2, 2],
                'B': [3, 4, 4],
                'v': [8, 9, 8]
            })
            PostProcessing.detect_dups(df=df, group_cols=['A', 'B'])

        self.assertRaises(ValueError, contains_dups)


if __name__ == '__main__':
    unittest.main()
