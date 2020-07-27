# coding: utf-8
import logging
import unittest

import pandas as pd

from cfg.paths import LOG_FILENAME
from src.tradeflow.TradeflowDC import TradeflowDC
from src.tradeflow.TradeflowIL import TradeflowIL
from src.utils.logging import setup_loggers

logger = logging.getLogger(__name__)
setup_loggers('test_tradeflow', log_filepath=LOG_FILENAME, level=logging.INFO)


class TestTradeflow(unittest.TestCase):

    def test_offtake_setter(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake.copy()
        tradeflow.ic_r = ic_r.copy()
        tradeflow.ic_s = ic_s.copy()
        tradeflow.update_from_offtake()

        new_offtake = pd.Series([10, 8, 9, 10, 11, 11, 9, 9, 8, 9, 14, 10])
        tradeflow.offtake = new_offtake.copy()
        tradeflow.update_from_offtake()

        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))

        logger.info('[OK] update_from_offtake')

    def test_offtake_setter_drop(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 0.001, 0.001, 0.001, 0.001, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r.copy()
        tradeflow.ic_s = ic_s.copy()
        tradeflow.update_from_offtake()
        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        logger.info('[OK] update_from_offtake')

    def test_offtake_setter_raise(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 10000, 10000, 10000, 10000, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r.copy()
        tradeflow.ic_s = ic_s.copy()
        tradeflow.update_from_offtake()
        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        logger.info('[OK] update_from_offtake')

    def test_sellout_setter(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 0.01, 0.01, 0.01, 0.01, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r.copy()
        tradeflow.ic_s = ic_s.copy()
        tradeflow.update_from_offtake()
        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        old_sellin = tradeflow.sellin
        tradeflow.sellout = 1.1 * tradeflow.sellout.copy(deep=True)
        tradeflow.update_from_sellout()
        self.assertTrue(offtake.equals(tradeflow.offtake))
        self.assertFalse(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        self.assertFalse(old_sellin.equals(tradeflow.sellin))
        logger.info('[OK] update_from_sellout')

    def test_sellout_setter_drop(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r
        tradeflow.ic_s = ic_s
        tradeflow.update_from_offtake()
        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        old_sellin = tradeflow.sellin

        new_sellout = tradeflow.sellout.copy(deep=True)
        new_sellout.iloc[6:10] = 0.01  # drop
        tradeflow.sellout = new_sellout
        tradeflow.update_from_sellout()
        self.assertTrue(offtake.equals(tradeflow.offtake))
        self.assertFalse(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        self.assertFalse(old_sellin.equals(tradeflow.sellin))
        logger.info('[OK] update_from_sellout')

    def test_sellout_setter_raise(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r
        tradeflow.ic_s = ic_s
        tradeflow.update_from_offtake()
        self.assertTrue(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        old_sellin = tradeflow.sellin

        new_sellout = tradeflow.sellout.copy(deep=True)
        new_sellout.iloc[6:10] = 100  # raise
        tradeflow.sellout = new_sellout
        tradeflow.update_from_sellout()
        self.assertTrue(offtake.equals(tradeflow.offtake))
        self.assertFalse(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(ic_s.equals(tradeflow.ic_s))
        self.assertFalse(old_sellin.equals(tradeflow.sellin))
        logger.info('[OK] update_from_sellout')

    def test_sellin_setter(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r
        tradeflow.ic_s = ic_s
        tradeflow.update_from_offtake()
        new_sellin = 1.1 * tradeflow.sellin.copy(deep=True)
        tradeflow.sellin = new_sellin
        tradeflow.update_from_sellin_constrained()
        logger.info('[OK] update_from_sellin')

    def test_sellin_setter_drop(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r
        tradeflow.ic_s = ic_s
        tradeflow.update_from_offtake()
        new_sellin = tradeflow.sellin.copy(deep=True)
        new_sellin.iloc[6:10] = 0.01  # drop
        tradeflow.sellin = new_sellin
        tradeflow.update_from_sellin_constrained()
        logger.info('[OK] update_from_sellin')

    def test_sellin_setter_raise(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        ic_r = pd.Series([0.5] * offtake.size)
        ic_s = pd.Series([1.0] * offtake.size)
        tradeflow = TradeflowDC()
        tradeflow.offtake = offtake
        tradeflow.ic_r = ic_r
        tradeflow.ic_s = ic_s
        tradeflow.update_from_offtake()
        new_sellin = tradeflow.sellin.copy(deep=True)
        new_sellin.iloc[6:10] = 10000  # raise
        tradeflow.sellin = new_sellin
        tradeflow.update_from_sellin_constrained()
        logger.info('[OK] update_from_sellin')

    def test_trackable(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        trackable_offtake = pd.Series([i + 1 for i in [10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10]])
        tradeflow = TradeflowIL()
        tradeflow.init_trackable(trackable_offtake=trackable_offtake, offtake=offtake)
        tradeflow.offtake = offtake.multiply(2)
        tradeflow.trackable_offtake = tradeflow.trackable_offtake.multiply(0.5)
        self.assertTrue(tradeflow.offtake.equals(offtake))
        tradeflow.trackable_offtake.astype(float).equals(trackable_offtake.astype(float))
        logger.info('[OK] trackable')

    def test_offtake_setter_eib(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        trackable_offtake = pd.Series([i + 1 for i in [10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10]])
        ic_r = pd.Series([0.5] * offtake.size)
        tradeflow = TradeflowIL()
        tradeflow.init_trackable(trackable_offtake=trackable_offtake, offtake=offtake)
        tradeflow.offtake = offtake.copy()
        tradeflow.ic_r = ic_r.copy()
        tradeflow.update_from_offtake()

        new_offtake = pd.Series([10, 8, 9, 10, 11, 11, 9, 9, 8, 9, 14, 10])
        tradeflow.offtake = new_offtake.copy()
        tradeflow.update_from_offtake()

        self.assertTrue(ic_r.equals(tradeflow.ic_r))

        logger.info('[OK] update_from_offtake')

    def test_sellin_setter_eib(self):
        offtake = pd.Series([10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10])
        trackable_offtake = pd.Series([i + 1 for i in [10, 8, 9, 10, 10, 11, 9, 9, 8, 9, 14, 10]])
        ic_r = pd.Series([0.5] * offtake.size)
        tradeflow = TradeflowIL()
        tradeflow.init_trackable(trackable_offtake=trackable_offtake, offtake=offtake)
        tradeflow.offtake = offtake.copy()
        tradeflow.ic_r = ic_r.copy()
        tradeflow.update_from_offtake()

        tradeflow.sellin = 1.1 * tradeflow.sellin.copy(deep=True)
        tradeflow.update_from_sellin()

        self.assertFalse(ic_r.equals(tradeflow.ic_r))
        self.assertTrue(offtake.equals(tradeflow.offtake))

        logger.info('[OK] update_from_sellin')


if __name__ == '__main__':
    unittest.main()
