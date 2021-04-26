import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Train pull backed 132 data points
# mydata = quandl.get("NSE/NIFTY_50", start_date='2011-01-01', end_date='2016-04-20')
# Test pull backed 132 data points
# mydata = quandl.get("NSE/NIFTY_50", start_date='2016-04-21', end_date='2016-10-31')

# Full
# mydata = mydata.drop(columns=['Turnover (Rs. Cr)'], axis=1).rename(columns={"Shares Traded": "Volume"})
# mydata = mydata.round(0)
# mydata = mydata.fillna(method='pad')
# mydata = mydata.dropna(axis=0)
# scaler = StandardScaler()
# mydata_scale = scaler.fit_transform(mydata)

# print(len(mydata_scale))
# print(mydata_scale)
# print(mydata_scale[:, 3])


# plot_data = mydata_scale[:, 3]
# fig, ax = plt.subplots(figsize=(30,10))
# title = str(i) + ' - ' + str(i+132) 
# ax.set_title(title)
# time = range(len(plot_data))
# ax.plot(time, plot_data, color='tab:red')
# ax.set_xlabel('time')
# ax.set_ylabel('stock price ($)')
# ax.set_xticks(np.arange(0, 140, 10))
# ax.set_xlim(0, 140)
# plt.savefig(f'./imgs/Search/{title}.png')
# plt.show()

# for i in range(0, len(mydata_scale), 132):
#         plot_data = mydata_scale[i:i+132, 3]
#         fig, ax = plt.subplots(figsize=(10,10))
#         title = str(i) + ' - ' + str(i+132) 
#         ax.set_title(title)
#         time = range(len(plot_data))
#         ax.plot(time, plot_data, color='tab:red')
#         ax.set_xlabel('time')
#         ax.set_ylabel('stock price ($)')
#         ax.set_xticks(np.arange(0, 140, 10))
#         ax.set_xlim(0, 140)
#         # plt.savefig(f'./imgs/Search/{title}.png')
#         plt.show()




data_columns =['High', 'Low']

d = 22
train_vals = mydata[data_columns].values
y_vals = mydata['Close'].values
print(train_vals)

print()
y_data = []

x_data = train_vals[d-d:d, :]
print(x_data)
print()
y_data.append(y_vals[0])
y_data.append(y_vals[d])
print(y_data)
print()
y_data = np.array(y_data)
print(y_data)
print()
y_data = y_data.reshape(1, -1)
print(y_data)
print()
y_data = np.tile(y_data.transpose(), (1, 1))
print(y_data)
print()

# tile(array([[1,2,3]]).transpose(), (1, 3))
# array([[1, 1, 1],
#        [2, 2, 2],
#        [3, 3, 3]])


# x_train, y_train = [], []
# for i in range(d, len(train_vals)):
#     x_train.append(train_vals[i-d:i, 0])
#     y_train.append(train_vals[i, 0])

# x_train, y_train = np.array(x_train), np.array(y_train)

# print(x_train.shape)
# print(x_train)
# print()

# x_train = np.reshape(x_train, (*x_train.shape, 1))

# print(x_train.shape)
# print(x_train)
# print()