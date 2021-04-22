# 

import quandl


        #     {'ticker':'NSE/NIFTY_50', 'start':'2011-01-01', 'end':''},
        # 'test':
        #     {'ticker':'NSE/NIFTY_50', 'start':'2016-06-21', 'end':'2016-12-31'}

mydata = quandl.get("NSE/NIFTY_50", start_date='2011-01-01', end_date='2016-04-20')
print(mydata.shape)

mydata = quandl.get("NSE/NIFTY_50", start_date='2016-04-21', end_date='2016-10-31')
print(mydata.shape)


# mydata = mydata.drop(columns=['Turnover (Rs. Cr)'], axis=1).rename(columns={"Shares Traded": "Volume"})
# mydata = mydata.round(0)
# mydata = mydata.fillna(method='pad')
# mydata = mydata.dropna(axis=0)

