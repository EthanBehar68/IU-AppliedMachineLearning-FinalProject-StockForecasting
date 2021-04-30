from lstm_behar import LSTM_Behar
from fastquant import backtest, get_stock_data
import pickle
import json
from test import *
from lstm_behar import *
from forecasting_train_predictor import *



# want to train on this data, need access to gaussian_hmm file
#df = get_stock_data('AAPL', '2020-01-01','2020-12-31')


def percentage_gain(init, end):
    return (end-init)/init


training = {
    'AAPL': {
        'start': '2017-08-01',
        'end': '2019-01-01'
    },
    'IBM': {
        'start': '2017-08-01',
        'end': '2019-01-01'
    },
    'AMZN': {
        'start': '2017-08-01',
        'end': '2019-01-01'
    },
    'MSFT': {
        'start': '2017-08-01',
        'end': '2019-01-01'
    },
    'GOOGL': {
        'start': '2017-08-01',
        'end': '2019-01-01'
    }
}


testing = {
    'AAPL': {
        'start': '2019-01-01',
        'end': '2020-05-31'
    },
    'IBM': {
        'start': '2019-01-01',
        'end': '2020-05-31'
    },
    'AMZN': {
        'start': '2019-01-01',
        'end': '2020-05-31'
    },
    'MSFT': {
        'start': '2019-01-01',
        'end': '2020-05-31'
    },
    'GOOGL': {
        'start': '2019-01-01',
        'end': '2020-05-31'
    }
}


params = {
    'lr': 0.001,
    'loss': 'mean_absolute_percentage_error', 
    'activation': 'tanh',
    'recurrent_activation': 'sigmoid',
    'epochs': 500,
    'batch_size': 150,
    'd': 22,
    'train_columns': ['high', 'low', 'open', 'close'],
    'label_column': 'close', 
    'name': 'Behar-Std-500-HighLowOpenClose',
    'discretization': False,
    'fill_method': 'previous',
    'normalization': False
}



behar = LSTM_Behar(params=params)
results = {}


for ticker,_ in zip(training.keys(), testing.keys()):
    print(ticker)
    print('getting stock data')
    start = training[ticker]['start']
    end = training[ticker]['end']
    training_data = get_stock_data(ticker, start, end)

    start = testing[ticker]['start']
    end = testing[ticker]['end']
    back_test_data = get_stock_data(ticker, start, end)

    print('training...')
    model = behar.gen_model()
    model.compile(optimizer='adam', loss='mse')
    #print(type(training_data))
    #print(training_data.info())
    #print(training_data.head())
    training_data = training_data['close']
    model.fit(training_data)
    preds, actual = behar.predict(test_data=back_test_data)

    preds = pd.DataFrame(data=np.array(preds), columns=['yhat'])
    expected_1day_return = preds['yhat'].pct_change().shift(-1).multiply(100)
    back_test_data['custom'] = expected_1day_return.multiply(-1).values

    print('testing...')
    res,history = backtest("custom",
                    back_test_data.dropna(),
                    upper_limit=3,
                    lower_limit=-2,
                    buy_prop=1,
                    sell_prop=1,
                    return_history=True,
                    execution_type='close',
                    plot=False,
                    verbose=0)



    print(res.info())

    results[f'{ticker}, {start}-{end}'] = percentage_gain(float(res['init_cash']), float(res['final_value']))


dump = json.dumps(results)
output_file = open('fastquant_results.json', 'w')
output_file.write(dump)
output_file.close()




if __name__ == "__main__":
    # ['high', 'low', 'open', 'close'] Test
    # Naming syntax please use
    # {Paper}-{Std/Norm}-{Discr/''}-{epoch}-{train columns}-{Rolling/Fixed}
    params = {'lr': 0.001,
                'loss': 'mean_absolute_percentage_error', 
                'activation': 'tanh',
                'recurrent_activation': 'sigmoid',
                'epochs': 500,
                'batch_size': 150,
                'd': 22,
                'train_columns': ['high', 'low', 'open', 'close'],
                'label_column': 'close', 
                'name': 'Behar-Std-500-HighLowOpenClose',
                'discretization': False,
                'fill_method': 'previous',
                'normalization': False}
    
    test = Test(Model=LSTM_Behar, Train_Predictor=Forecasting_Train_Predictor, params=params, tests=window_heavy_hitters_tests, plot=True)
    # Make sure this folder is create or MatLibPlot will error out!!
    test.rolling_window_test('./forecasting_rnns/results/Behar-Std-500-HighLowOpenClose/')