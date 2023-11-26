import numpy as np
import pandas as pd

from Model.model import *
from Preprocess.preprocess import *

'''
In our code, the provided training dataset will be referred to as the "training data",
the public leaderboard dataset will be referred to as the "validation data", and the
private leaderboard dataset will be reffered to as the "testing data".
'''

# Load Data
print('Loading Data ...')
data, train_txkeys, val_txkeys, y_public = load_data()


# Preprocess Data
print('Preprocessing Data ...')
data = generate_others(data)
data = handle_missing(data, val_txkeys, y_public)


# Generate New Features
print('Generating New Features ...')
data = generate_features(data)


# Prepare Data for Training
print('Preparing Data for Training ...')
X_train, y_train, X_val, y_val, X_test, val_txkeys, test_txkeys = prepare_data(data, train_txkeys)
cat_cols, X_train_cb, X_val_cb, X_test_cb = prepare_data_cb(X_train, X_val, X_test)


# Hyperparameter Tuning
perform_tuning = False  # set to True if you want to perform hyperparameter tuning
if perform_tuning == True:
    print('Performing Hyperparameter Tuning ...')
    tune_parameters(X_train, X_train_cb, y_train, cat_cols, X_val, X_val_cb, y_val)


# Train and Make Predictions

# For Public Leaderboard
# Use training data only for training
print('Training for Public Leaderboard ...')
xgb_model, lgb_model, cb_model = train(X_train, X_train_cb, y_train, cat_cols)
print('Predicting for Public Leaderboard ...')
y_pred_val_xgb, y_pred_val_lgb, y_pred_val_cb = predict(xgb_model, lgb_model, cb_model, X_val, X_val_cb)

# For Private Leaderboard
# Use training data and validation data for training
X_merged = pd.concat([X_train, X_val], ignore_index=True)
y_merged = pd.concat([y_train, y_val], ignore_index=True)
X_merged_cb = pd.concat([X_train_cb, X_val_cb], ignore_index=True)
del X_train, y_train, X_val, y_val, X_train_cb, X_val_cb

print('Training for Private Leaderboard ...')
xgb_model, lgb_model, cb_model = train(X_merged, X_merged_cb, y_merged, cat_cols)
print('Predicting for Private Leaderboard ...')
y_pred_test_xgb, y_pred_test_lgb, y_pred_test_cb = predict(xgb_model, lgb_model, cb_model, X_test, X_test_cb)


# Make Final Predictions
print('Making Final Predictions ...')
make_submission(val_txkeys, test_txkeys, y_pred_val_xgb, y_pred_test_xgb, y_pred_val_lgb, y_pred_test_lgb, y_pred_val_cb, y_pred_test_cb)


print('Done.')