from lstm_behar import LSTM_Behar
from fastquant import backtest, get_stock_data
import pickle
import json
from test import *
from lstm_pawar import *
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



pawar = LSTM_Pawar(params=params)
results = {}


for ticker,_ in zip(training.keys(), testing.keys()):
    train_predictor = Forecasting_Train_Predictor(params=params)
    model = pawar.gen_model()

    print(ticker)
    print('getting stock data')
    start = training[ticker]['start']
    end = training[ticker]['end']
    training_data = pawar.get_data(ticker, start, end)

    start = testing[ticker]['start']
    end = testing[ticker]['end']
    back_test_data = pawar.get_data(ticker, start, end)


    print('training...')
    #print(type(training_data))
    #print(training_data.info())
    #print(training_data.head())
    model, _ = train_predictor.train(model, training_data, pawar.label_column_index)
    preds, actual = train_predictor.predict(model, back_test_data, pawar.label_column_index)

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




    results[f'{ticker}, {start}-{end}'] = percentage_gain(float(res['init_cash']), float(res['final_value']))


dump = json.dumps(results)
output_file = open('pawar_fastquant_results.json', 'w')
output_file.write(dump)
output_file.close()