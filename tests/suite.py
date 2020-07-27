# coding: utf-8
import unittest

from tests.test_export_to_apo import TestExportToAPO
from tests.test_tradeflow import TestTradeflow


def suite():
    suite_forecast = unittest.TestSuite()

    suite_forecast.addTests(  # todo Thibaut: complete
        tests=[
            TestTradeflow('test_offtake_setter'),
            TestTradeflow('test_inventory_coverage_setter'),
            TestTradeflow('test_sellin_setter'),
            TestExportToAPO('test_weeks_in_month'),
        ]
    )

    return suite_forecast


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
