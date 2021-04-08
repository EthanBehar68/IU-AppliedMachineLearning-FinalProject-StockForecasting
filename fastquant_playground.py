from fastquant import get_stock_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = get_stock_data("AAPL", "2001-03-26", "2021-03-26")
print(df['volume']/df['volume'].sum())