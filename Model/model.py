import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

def prepare_data(data, train_txkeys):
    
    data_test = data[data['label'].isna()]
    data = data[~data['label'].isna()]

    data_public_old = pd.read_csv('./data/public_processed.csv')
    val_txkeys = np.unique(data_public_old['txkey'])

    data_val = data[data['txkey'].isin(val_txkeys)]
    data_train = data[data['txkey'].isin(train_txkeys)]

    del data, train_txkeys

    id_features = ['txkey', 'chid', 'cano', 'mchno', 'acqic']
    categorical_features = ['txkey', 'chid', 'cano', 'contp', 'etymd', 'mchno', 'acqic', 'mcc', 'stocn', 'scity', 'stscd', 'hcefg', 'csmcu']
    numerical_features = ['locdt', 'loctm', 'conam', 'iterm', 'flam1', 'csmam']
    bool_features = ['ecfg', 'insfg', 'bnsfg', 'ovrlt', 'flbmk', 'flg_3dsmk']
    cat_without_id = ['contp', 'etymd', 'mcc', 'stocn', 'scity', 'stscd', 'hcefg', 'csmcu']

    val_txkeys = data_val['txkey']
    test_txkeys = data_test['txkey']

    X_train = data_train.drop(id_features, axis=1).drop('label', axis=1)
    y_train = data_train.drop(id_features, axis=1)['label']
    X_val = data_val.drop(id_features, axis=1).drop('label', axis=1)
    y_val = data_val.drop(id_features, axis=1)['label']
    X_test = data_test.drop(id_features, axis=1).drop('label', axis=1)

    del data_train, data_val, data_test

    return X_train, y_train, X_val, y_val, X_test, val_txkeys, test_txkeys


def prepare_data_cb(X_train, X_val, X_test):
    
    cat_cols = ['contp', 'etymd', 'mcc', 'stocn', 'scity', 'stscd', 'hcefg', 'csmcu']

    X_train_cb = X_train.copy()
    X_train_cb[cat_cols] = X_train_cb[cat_cols].astype('int')
    X_train_cb[cat_cols] = X_train_cb[cat_cols].astype('category')

    X_val_cb = X_val.copy()
    X_val_cb['mcc'] = X_val_cb['mcc'].cat.add_categories([-888])
    X_val_cb['mcc'] = X_val_cb['mcc'].fillna(-888)
    X_val_cb['stocn'] = X_val_cb['stocn'].cat.add_categories([-888])
    X_val_cb['stocn'] = X_val_cb['stocn'].fillna(-888)
    X_val_cb['scity'] = X_val_cb['scity'].cat.add_categories([-888])
    X_val_cb['scity'] = X_val_cb['scity'].fillna(-888)
    X_val_cb['next_card_min'] = X_val_cb['next_card_min'].fillna(-888)

    X_val_cb[cat_cols] = X_val_cb[cat_cols].astype('int')
    X_val_cb[cat_cols] = X_val_cb[cat_cols].astype('category')

    X_test_cb = X_test.copy()
    X_test_cb['mcc'] = X_test_cb['mcc'].cat.add_categories([-888])
    X_test_cb['mcc'] = X_test_cb['mcc'].fillna(-888)
    X_test_cb['stocn'] = X_test_cb['stocn'].cat.add_categories([-888])
    X_test_cb['stocn'] = X_test_cb['stocn'].fillna(-888)
    X_test_cb['scity'] = X_test_cb['scity'].cat.add_categories([-888])
    X_test_cb['scity'] = X_test_cb['scity'].fillna(-888)
    X_test_cb['next_card_min'] = X_test_cb['next_card_min'].fillna(-888)

    X_test_cb[cat_cols] = X_test_cb[cat_cols].astype('int')
    X_test_cb[cat_cols] = X_test_cb[cat_cols].astype('category')

    return cat_cols, X_train_cb, X_val_cb, X_test_cb


def tune_parameters(X_train, X_train_cb, y_train, cat_cols):
    
    # XGBoost
    param_grid = {
        'n_estimators': [100, 300, 600],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2, 0.3],
        'reg_lambda': [0, 10, 20],
    }

    optimal_params = GridSearchCV(estimator=xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True),
                                param_grid=param_grid,
                                cv=StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=0),
                                scoring='f1',
                                return_train_score=True
                                ).fit(X_train, y_train)

    print("Best parameters for XGB Model:", optimal_params.best_params_)
    print("Score using the best parameters:", optimal_params.best_score_)

    # LightGBM
    param_grid = [
        {
        'max_depth': [6],
        'num_leaves': [2**6-1],
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.05, 0.1, 0.2],
        'reg_lambda': [0, 10, 20],
        },
        {
        'max_depth': [8],
        'num_leaves': [2**8-1],
        'n_estimators': [200, 400, 600],
        'learning_rate': [0.05, 0.1, 0.2],
        'reg_lambda': [0, 10, 20],
        },
    ]

    optimal_params = GridSearchCV(estimator=lgb.LGBMClassifier(objective='binary', verbosity=-1),
                                param_grid=param_grid,
                                cv=StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=0),
                                scoring='f1',
                                return_train_score=True
                                ).fit(X_train, y_train)

    print("Best parameters for LGB Model:", optimal_params.best_params_)
    print("Score using the best parameters:", optimal_params.best_score_)

    # CatBoost
    param_grid = {
        'iterations': [600, 800, 1000],
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.04, 0.1],
        'l2_leaf_reg': [0, 10, 20]
    }

    optimal_params = GridSearchCV(estimator=CatBoostClassifier(cat_features=cat_cols, boosting_type='Plain', leaf_estimation_iterations=1, one_hot_max_size=254, silent=True),
                                param_grid=param_grid,
                                cv=StratifiedShuffleSplit(test_size=0.2, n_splits=1, random_state=0),
                                scoring='f1',
                                return_train_score=True
                                ).fit(X_train_cb, y_train)

    print("Best parameters for CB Model:", optimal_params.best_params_)
    print("Score using the best parameters:", optimal_params.best_score_)


def train(X_train, X_train_cb, y_train, cat_cols):

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', enable_categorical=True,
                                reg_lambda=20,
                                learning_rate=0.05,
                                max_depth=4,
                                n_estimators=600)
    xgb_model.fit(X_train, y_train)

    lgb_model = lgb.LGBMClassifier(objective='binary',
                                verbosity=-1,
                                n_estimators=400,
                                learning_rate=0.05,
                                max_depth=6,
                                reg_lambda=20,
                                num_leaves=2**6-1)
    lgb_model.fit(X_train, y_train)

    cb_model = CatBoostClassifier(cat_features=cat_cols,
                                boosting_type='Plain',
                                leaf_estimation_iterations=1,
                                one_hot_max_size=254,
                                silent=True,
                                depth=8,
                                iterations=1000,
                                l2_leaf_reg=0,
                                learning_rate=0.04)
    cb_model.fit(X_train_cb, y_train)

    return xgb_model, lgb_model, cb_model


def predict(xgb_model, lgb_model, cb_model, X, X_cb):

    y_pred_xgb = xgb_model.predict(X)
    y_pred_lgb = lgb_model.predict(X)
    y_pred_cb = cb_model.predict(X_cb)

    return y_pred_xgb, y_pred_lgb, y_pred_cb


def make_submission(val_txkeys, test_txkeys, y_pred_val_xgb, y_pred_test_xgb, y_pred_val_lgb, y_pred_test_lgb, y_pred_val_cb, y_pred_test_cb):

    txkey_total = np.concatenate((val_txkeys, test_txkeys), axis=0)

    xgb_pred_total = np.concatenate((y_pred_val_xgb, y_pred_test_xgb), axis=0)
    lgb_pred_total = np.concatenate((y_pred_val_lgb, y_pred_test_lgb), axis=0)
    cb_pred_total = np.concatenate((y_pred_val_cb, y_pred_test_cb), axis=0)

    pred_total = {'xgb': xgb_pred_total,
                  'lgb': lgb_pred_total,
                  'cb': cb_pred_total}
    pred_total = pd.DataFrame(pred_total)
    pred_total = pred_total.mode(axis=1, dropna=True).iloc[:, 0]

    submission = pd.DataFrame({'txkey': txkey_total, 'pred': pred_total})
    submission = submission.astype({'pred': 'int'})
    submission.to_csv('./submission.csv', index=False)