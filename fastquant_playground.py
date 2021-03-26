from fastquant import get_stock_data
import pandas as pd
import numpy as np

df = get_stock_data("AAPL", "2020-01-01", "2021-01-01")
print(df.head())