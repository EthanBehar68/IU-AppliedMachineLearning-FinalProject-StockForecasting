# P556FinalProject

## Notes for Dr. Williamson and Junyi

Work for Section 3.1 "Models for Fractional Change Prediction" can be found in frac_change_forecasting.

Work for Section 3.2 "Models for Raw Close Price Prediction" can be found in raw_close_forecasting_rnns.  
  - Within this folder:  
    - lstm_X.py contain the models we built.  
    - X_train_predictor.py contain the train/predict methods used on the models.  
    - test.py is the main driving code for running and testing our models  
    - Any other *test*.py files are entry points for test.py.  
    - Playgound.py is a file used to test random bits of code and can be ignored.  
    - The old folder can be ignored. It contains early works that we've moved away from.  
    - The results folder contains the graphs for each of the tests organized by specific test.  
      - The excessive folder contains many more graphs of tests we ran but realized was unnecessary.  

Work for our sentiment anaylsis proof of concept can be found in sentiment.

## requirements

Make sure you are using python3.6 (highest version of python tensorflow works with). Create a virtual environment by running:

```
python3.6 -m venv proj_env
```

Once the environment is created run:

```
source proj_env/bin/activate
```

Then go ahead an install the requirements:

```
python3.6 -m pip install --upgrade pip
python3.6 -m pip install -r requirements.txt
```

To run it with a jupyter notebook:

```
python3.6 -m ipykernel install --user --name=proj_env
```
