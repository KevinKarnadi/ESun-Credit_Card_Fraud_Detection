# Model Module

## Overview
This module provides code for preparing data for the model, hyparameter tuning by GridSearchCV, and for training and predicting using the models developed.

## Functions

### prepare_data() and prepare_data_cb()
Prepare data to be used for training and predicting using the models.

### tune_parameters()
Perform hyperparameter tuning on the three different models: XGBoost, LightGBM, and CatBoost using GridSearchCV.

### train()
Train the three different models by fitting it into a dataset.

### predict()
Make predictions using the three different models.

### make_submission()
Make the final ensembled prediction by performing max voting on the predictions from each model.

## Parameter Settings

### XGBoost
- reg_lambda: 20
- learning_rate: 0.05
- max_depth: 4
- n_estimators: 600

### LightGBM
- n_estimators: 400
- learning_rate: 0.05
- max_depth: 6
- reg_lambda: 20
- num_leaves: 63

### CatBoost
- depth: 8
- iterations: 1000
- l2_leaf_reg: 0
- learning_rate: 0.04
- boosting_type: 'Plain'
- leaf_estimation_iterations: 1
- one_hot_max_size: 254
