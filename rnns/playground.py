# 

import quandl

mydata = quandl.get("NSE/NIFTY_50", start_date='2011-01-01', end_date='2016-12-31')

print(type(mydata))

mydata = mydata.drop(columns=['Turnover (Rs. Cr)'], axis=1).rename(columns={"Shares Traded": "Volume"})
mydata = mydata.round(0)
mydata = mydata.fillna(method='pad')


print(mydata)
