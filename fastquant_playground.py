from fastquant import get_stock_data, backtest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = get_stock_data("AAPL", "2001-03-26", "2021-03-26")
#print(df.head())
#print(df['open'][0])



res = backtest("smac", df, fast_period=range(15,30,3), slow_period=range(40,55,3), verbose=False)
print(res[['fast_period', 'slow_period', 'final_value']].head())
