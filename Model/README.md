# Model Module

## Overview
This module provides code for preparing data for the model, hyparameter tuning by GridSearchCV, and for training and predicting using the models developed.

## Functions

### prepare_data() and prepare_data_cb()
Prepare data to be used for training and predicting using the models.
- Split data into training, validation, and testing
- Split features and labels

### tune_parameters()
Set the ```perform_tuning``` variable in ```main.py``` to ```True``` if you want to perform hyperparameter tuning.

For the three different models: XGBoost, LightGBM, and CatBoost, perform hyperparameter tuning using an exhaustive grid search to find the best combination of parameters for each model. The parameters that is tried to optimize for here are the ```n_trees```, ```learning rate```, ```max depth``` (which increases the complexity of the model), and the ```lambda``` parameter (which is for regularization)

- XGBoost parameters searched over:
  - ```n_estimators```: 100, 300, 600
  - ```learning_rate```: 0.05, 0.1, 0.2, 0.3
  - ```max_depth```: 4, 6, 8
  - ```reg_lambda```: 0, 10, 20
- LightGBM parameters searched over:
  - ```n_estimators```: 200, 400, 600
  - ```learning_rate```: 0.05, 0.1, 0.2
  - ```max_depth```: 6, 8
  - ```reg_lambda```: 0, 10, 20
- CatBoost parameters searched over:
  - ```iterations```: 600, 800, 1000
  - ```learning_rate```: 0.01, 0.04, 0.1
  - ```depth```: 4, 6, 8
  - ```l2_leaf_reg```: 0, 10, 20

The parameters are tuned using a 20% subset of the training data, and the final performance of the model will be assessed using the holdout validation data which contains the public leaderboard data.

### train()
Train the three different models by fitting it into a dataset. To make predictions for the public leaderboard, the models are only trained on the training data, while to make predictions for the private leaderboard, the models are trained on the training data + the public leaderboard data with their given labels.

### predict()
Make predictions using the three different models. There are two predictions that will be made here: one for the public leaderboard data, and one for the private leaderboard data.

### make_submission()
Make the final ensembled prediction by performing max voting on the predictions from each model.

## Parameter Settings

### XGBoost
- ```reg_lambda```: 20
- ```learning_rate```: 0.05
- ```max_depth```: 4
- ```n_estimators```: 600

### LightGBM
- ```n_estimators```: 400
- ```learning_rate```: 0.05
- ```max_depth```: 6
- ```reg_lambda```: 20
- ```num_leaves```: 63

### CatBoost
- ```depth```: 8
- ```iterations```: 1000
- ```l2_leaf_reg```: 0
- ```learning_rate```: 0.04
- ```boosting_type```: 'Plain'
- ```leaf_estimation_iterations```: 1
- ```one_hot_max_size```: 254
