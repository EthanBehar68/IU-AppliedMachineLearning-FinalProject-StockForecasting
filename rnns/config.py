LSTMparams = {
    "time_frame": 365,
    "ticker": "ETH-USD",
    "testing_days": 60
}
GRUparams = {
    "train_start":"2013-01-01",
    "train_end":"2019-12-31",
    "test_start":"2020-01-01",
    "test_end":"2020-12-31",
    "ticker": "TSLA",
    "inputs": ['open','high','low','close','volume'],
    "previous_days": 60,
    "optimizer": "rmsprop",
    "loss": 'mean_squared_error',
    "epochs": 100,
    "batch_size":150
}
