from fastquant import backtest, get_stock_data
import pickle
from gaussian_hmm import *
import json
#from forecasting_train_predictor import *
#from overfit_train_predictor import *

# want to train on this data, need access to gaussian_hmm file
df = get_stock_data('AAPL', '2020-01-01','2020-12-31')


params = {
    'n_components': 2,
    'algorithm': 'map',
    'n_iter': 100,
    'd': 5,
    'name':'GHMM'
}


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





ghmm = GHMM(params=params)
results = {}


for ticker,_ in zip(training.keys(), testing.keys()):
    print(ticker)
    print('getting stock data')
    
    # pull in the data
    start = training[ticker]['start']
    end = training[ticker]['end']
    training_data = get_stock_data(ticker, start, end)
    #training_data = training_data.drop(['volume'], axis=1)

    # create df for back testing
    start = testing[ticker]['start']
    end = testing[ticker]['end']
    back_test_data = get_stock_data(ticker, start, end)

    print('training...')
    ghmm.train(train_data=training_data)
    preds, actual = ghmm.predict(test_data=back_test_data)

    # get predictions, change them to percent changes and inverse them
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
                    #plot=False,
                    verbose=0)



    print(res.info())

    results[f'{ticker}, {start}-{end}'] = percentage_gain(float(res['init_cash']), float(res['final_value']))


dump = json.dumps(results)
output_file = open('fastquant_results.json', 'w')
output_file.write(dump)
output_file.close()
