import numpy as np
import pandas as pd

def load_data():
    
    data_train = pd.read_csv('./data/training.csv')
    train_txkeys = np.unique(data_train['txkey'])

    data_public = pd.read_csv('./data/public.csv')
    val_txkeys = np.unique(data_public['txkey'])
    y_public = data_public['label'].tolist()
    data_public = data_public.drop(['label'], axis=1)

    data_private = pd.read_csv('./data/private_1_processed.csv')

    data = pd.concat([data_train, data_public, data_private], ignore_index=True)

    data = data.astype({'txkey': 'category',
                        'locdt': 'float32',
                        'loctm': 'float32',
                        'chid': 'category',
                        'cano': 'category',
                        'contp': 'category',
                        'etymd': 'category',
                        'mchno': 'category',
                        'acqic': 'category',
                        'mcc': 'category',
                        'conam': 'float32',
                        'ecfg': 'bool',
                        'insfg': 'bool',
                        'iterm': 'float32',
                        'bnsfg': 'bool',
                        'flam1': 'float32',
                        'stocn': 'float32',
                        'scity': 'float32',
                        'stscd': 'category',
                        'ovrlt': 'bool',
                        'flbmk': 'bool',
                        'hcefg': 'category',
                        'csmcu': 'category',
                        'csmam': 'float32',
                        'flg_3dsmk': 'bool'})
    
    return data, train_txkeys, val_txkeys, y_public

def generate_others(data):

    threshold = 2000

    counts = data['mcc'].value_counts()
    other_mccs = counts[counts < threshold].index
    data['mcc'] = data['mcc'].apply(lambda x: -1 if x in other_mccs else x)
    data = data.astype({'mcc': 'category'})

    def generate_other_cities(data):
        total_count = data.groupby('stocn')['scity'].transform('count')
        mask = (data.groupby(['stocn', 'scity'])['scity'].transform('count') / total_count) < 0.2
        data.loc[mask, 'scity'] = -data.loc[mask, 'stocn'] - 1

    generate_other_cities(data)

    return data


def handle_missing(data, val_txkeys, y_public):

    data_valtest = data[data['label'].isna()]
    data = data[~data['label'].isna()]

    data = data.dropna(subset=['mcc', 'stocn'])
    data = pd.concat([data, data_valtest], ignore_index=True)

    data['stscd'] = data['stscd'].cat.add_categories([-1])
    data['stscd'] = data['stscd'].fillna(-1)

    data['hcefg'] = data['hcefg'].fillna(data['hcefg'].mode()[0])

    mode_by_stocn = data.groupby('stocn')['csmcu'].apply(lambda x: x.mode().iloc[0])
    data.loc[data['csmcu'].isna(), 'csmcu'] = data.loc[data['csmcu'].isna(), 'stocn'].map(mode_by_stocn)

    mode_by_stocn = data.groupby('stocn')['scity'].apply(lambda x: x.mode().iloc[0])
    data.loc[data['scity'].isna(), 'scity'] = data.loc[data['scity'].isna(), 'stocn'].map(mode_by_stocn)

    data = data.astype({'stocn': 'category', 'scity': 'category'})

    data.loc[data['etymd'].isna(), 'etymd'] = np.random.choice([4, 5, 8], size=data['etymd'].isna().sum())

    data_else = data[~data['txkey'].isin(val_txkeys)]
    data_val = data[data['txkey'].isin(val_txkeys)]

    data_val.loc[:, 'label'] = y_public

    data = pd.concat([data_else, data_val], ignore_index=True)

    del val_txkeys, y_public

    return data


def generate_features(data):

    data = data.sort_values(by = ['chid','cano','locdt','loctm']).reset_index(drop = True)

    # Transaction time features
    def time_to_period(time):
        if 0 <= time <= 55959:
            return 0  # Dawn
        elif 60000 <= time <= 95959:
            return 1  # Morning
        elif 100000 <= time <= 145959:
            return 2  # Afternoon
        elif 150000 <= time <= 195959:
            return 3  # Evening
        elif 200000 <= time <= 235959:
            return 4  # Night

    def date_to_week(date):
        return ((date-1) // 7 + 1)* 7

    data['loctm_period'] = data['loctm'].apply(time_to_period)
    data['week'] = data['locdt'].apply(date_to_week)

    # The difference in days (locdt) between two transactions associated with the same unique combination of card (cano) and merchant (mchno).
    data_head = data.groupby(['chid','mchno','week']).head(1)[['chid','mchno','week','locdt']]
    data_head = data_head.rename(columns = {'locdt' : 'locdt_head'})
    data_tail = data.groupby(['chid','mchno','week']).tail(1)[['chid','mchno','week','locdt']]
    data_tail = data_tail.rename(columns = {'locdt' : 'locdt_tail'})
    data_head = pd.merge(data_head, data_tail, how = 'left', on = ['chid','mchno','week'])
    data_head['same_chid_mchno_separate_trans_locdt_diff'] = data_head['locdt_tail'] - data_head['locdt_head']
    data = pd.merge(data,data_head, how = 'left', on =['chid','mchno','week'])

    # Group by 'chid', 'cano', and 'week' and calculate aggregated values
    data['same_chid_cano_week_max_locdt'] = data.groupby(['chid', 'cano', 'week'])['locdt'].transform('max')
    data['same_chid_cano_week_min_locdt'] = data.groupby(['chid', 'cano', 'week'])['locdt'].transform('min')

    # Sort by 'cano' and 'same_chid_cano_week_max_locdt'
    data = data.sort_values(by=['cano', 'same_chid_cano_week_max_locdt'])

    # Calculate 'next_card_min' using transform
    data['next_card_min'] = data.groupby(['chid', 'week'])['same_chid_cano_week_min_locdt'].shift(-1)
    data['next_card_min'] = np.where(data['same_chid_cano_week_max_locdt'] - data['next_card_min'] >= 0, np.nan, data['next_card_min'])
    data['same_chid_cano_week_separate_trans_locdt_diff'] = data['same_chid_cano_week_max_locdt'] - data['next_card_min']
    data['same_chid_cano_week_separate_trans_locdt_diff'] = data['same_chid_cano_week_separate_trans_locdt_diff'].fillna(0)

    # Rename columns
    data = data.rename(columns={'same_chid_cano_week_max_locdt': 'cano_last_trans_locdt'})

    # Calculate additional features
    data['diff_locdt_with_last_trans_cano'] = data['locdt'] - data['cano_last_trans_locdt']
    data['diff_locdt_with_last_trans_week_cano'] = data['week'] - data['cano_last_trans_locdt']

    # Xonvert loctm to senconds
    def loctm_to_global_time(data):

        data = data.copy()
        data['loctm'] = data['loctm'].astype(str)
        data['loctm'] = data['loctm'].str[:-2]
        data['hours'] = data['loctm'].str[-6:-4]
        data['hours'] = np.where(data['hours']=='', '0', data['hours']).astype(int)
        data['minutes'] = data['loctm'].str[-4:-2]
        data['minutes'] = np.where(data['minutes']=='', '0', data['minutes']).astype(int)
        data['second'] = data['loctm'].str[-2:].astype(int)
        data['loctm'] = data['hours']*60*60 + data['minutes']*60 + data['second']
        data['global_time'] = data['locdt']*24*60*60 + data['hours']*60*60+data['minutes']*60+data['second']

        return data['global_time']

    data['global_time'] = loctm_to_global_time(data)

    #  The time(loctm) difference between the transaction and previous transaction (in seconds)
    data['same_cano_week_separate_trans_diff'] = data.groupby(['cano','week'])['global_time'].diff(periods = 1)
    data['same_cano_week_separate_trans_diff'] = data['same_cano_week_separate_trans_diff'].fillna(0)

    # Number of unique cards owned per user
    data['number_of_unique_cards_owned'] = data.groupby('chid')['cano'].transform('nunique')

    # Number of mcc per card (cano)
    data['number_of_mcc_per_cano'] = data.groupby('cano')['mcc'].transform('nunique')

    # Frequency of mcc per card
    data['mcc_frequency_per_cano'] = data.groupby('cano')['mcc'].transform('count')

    # The number of transaction for the merchant (mchno)
    data['same_cano_mchno_separate_trade_number'] = data.groupby(['cano', 'mchno'])['txkey'].transform('count')

    # The number of transaction during the week
    data['same_cano_week_separate_trade_number'] = data.groupby(['cano', 'week'])['txkey'].transform('count')

    # The number of transaction per person during the week
    data['same_chid_week_separate_trade_number'] = data.groupby(['chid', 'week'])['txkey'].transform('count')

    # The minimum amount(conam) of transaction of the card(cano) during the day(locdt)
    data['same_cano_locdt_separate_conam_min'] = data.groupby(['cano', 'locdt'])['conam'].transform('min')

    # The maximum amount(conam) of transaction of the card(cano) during the day(locdt)
    data['same_cano_locdt_separate_conam_max'] = data.groupby(['cano', 'locdt'])['conam'].transform('max')

    # The mean amount(conam) of transaction of the card(cano) during the day(locdt)
    data['same_cano_locdt_separate_conam_mean'] = data.groupby(['cano', 'locdt'])['conam'].transform('mean')

    # The minimum amount(conam) of transaction of the card(cano) during the week
    data['same_cano_week_separate_conam_min'] = data.groupby(['cano', 'week'])['conam'].transform('min')

    # The maximum amount(conam) of transaction of the card(cano) during the week
    data['same_cano_week_separate_conam_max'] = data.groupby(['cano', 'week'])['conam'].transform('max')

    # The mean amount(conam) of transaction of the card(cano) during the week
    data['same_cano_week_separate_conam_mean'] = data.groupby(['cano', 'week'])['conam'].transform('mean')

    # The number of transactions of a unique card in a certain city
    data['same_cano_separate_city_count'] = data.groupby('cano')['scity'].transform('count')
    # The number of distinct cities where the card has been utilized
    data['same_cano_separate_city_nunique'] = data.groupby('cano')['scity'].transform('nunique')

    # The number of transactions of a unique card in a certain country
    data['same_cano_separate_country_count'] = data.groupby('cano')['stocn'].transform('count')
    # The number of distinct countries where the card has been utilized
    data['same_cano_separate_country_nunique'] = data.groupby('cano')['stocn'].transform('nunique')

    # The n-th transaction with same card(cano) and merchant(mchno)
    data['cano_mchno_index'] = 1
    data['cano_mchno_index'] = data.groupby(['cano', 'mchno'])['cano_mchno_index'].cumsum()

    return data