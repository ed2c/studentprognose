import unittest
from tests.test_sarima import TestSarima


class TestAll(unittest.TestCase):
    def test_run_all(self):
        self.test_sarima()

    def test_sarima(self):
        test_sarima = TestSarima()
        test_sarima.initialise()
        test_sarima.test_bedrijfskunde()
        test_sarima.test_ai()
        test_sarima.test_psychologie()
