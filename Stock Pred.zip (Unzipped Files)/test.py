import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from keras.models import load_model
from tensorflow.python.tools import module_util as _module_util
import streamlit as st 
import yfinance as yf
import unittest

def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def calculate_moving_averages(data):
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    return ma100, ma200

class TestStockPrediction(unittest.TestCase):

    def test_fetch_stock_data(self):
        # Test when valid data is fetched
        data = fetch_stock_data('AAPL', '2022-01-01', '2022-01-10')
        self.assertTrue(isinstance(data, pd.DataFrame))

        # Test when invalid data is fetched
        with self.assertRaises(Exception):
            fetch_stock_data('INVALID', '2022-01-01', '2022-01-10')

    def test_calculate_moving_averages(self):
        data = pd.DataFrame({'Close': [10, 20, 30, 40, 50]})
        ma100, ma200 = calculate_moving_averages(data)
        self.assertTrue(all(isinstance(ma, pd.Series) for ma in [ma100, ma200]))

if __name__ == "__main__":
    unittest.main()
