# Preprocessing Module
## Overview
This module provides a set of functions designed to prepare and enhance a transaction dataset. These functions include loading dataset, generating new categories for infrequent entries, handling missing values, and performing feature engineering.

## Functions
### 'load_data()'
This function is the starting point for preprocessing task. It loads the training, public, and private datasets, and returns a merged dataset along with the transaction keys for training and validation purposes.

### 'generate_others(data)'
This function identifies infrequent entries (counts fall below a defined treshold) in the 'mcc' and 'scity' columns and groups them into a new category in each column. This is particularly useful for handling rare occurrences and improving model robustness.

### 'handle_missing(data, val_txkeys, y_public)'
This function addresses missing values to ensure that the dataset is ready for further analysis. We handle missing values in the training data by:
- Dropping rows with missing values in critical columns.
- Imputeing missing values in specific columns using predefined values.
- Imputing missing values based on grouped statistics.

### 'generate_features(data)'
This function creates new features based on transaction time, transaction counts, and other relevant aspects which may contribute valuable information for model training.
New generated features are as follows:
#### 1. Time-based features
- 'week': categorizing the date (locdt) into weekly bins
- 'loctm_period': categorizing the time (loctm) into 5 different bins
- 'same_cano_week_separate_trans_diff': time difference (in seconds) between a transaction and the preceding transaction made with the same card (cano) within the same week
- 'diff_locdt_with_last_trans_cano': time difference (in days) between a transaction and the preceding transaction made with same card (cano) within the same week
- 'same_chid_cano_week_separate_trans_locdt_diff': time span (in days) between transactions of cards owned by the same individual (chid) universally (all-time)
#### 2. Transaction value features
- 'same_cano_locdt_separate_conam_min': lowest transaction values (conam) for an unique card (cano) within a given day (locdt)
- 'same_cano_locdt_separate_conam_max': highest transaction values (conam) for an unique card (cano) within a given day (locdt)
- 'same_cano_locdt_separate_conam_mean': average transaction values (conam) for an unique card (cano) within a given day (locdt)
- 'same_cano_week_separate_conam_min': lowest transaction values (conam) for an unique card (cano) within a given week
- 'same_cano_week_separate_conam_max': highest transaction values (conam) for an unique card (cano) within a given week
- 'same_cano_week_separate_conam_mean': mean transaction values (conam) for an unique card (cano) within a given week
#### 3. Merchant-related features
- 'same_chid_mchno_separate_trans_locdt_diff': the difference in days (locdt) between the first and most recent transaction associated with a specific combination of card (cano) and merchant (mchno)
- 'cano_mchno_index': the transaction sequence (n-th) number for a specific card (cano) and merchant (mchno) pairing
#### 4. Transaction frequency features
- 'same_cano_mchno_separate_trade_number': frequency of transaction of a unique card to a certain unique merchant store
- 'same_cano_week_separate_trade_number': frequency of transaction of a unique card in a given week
- 'same_chid_week_separate_trade_number': frequency of transaction of a unique individual in a given week
#### 5. Geographical features
- 'same_cano_separate_city_count': the number of transactions of a unique card in a certain city
- 'same_cano_separate_city_nunique': the number of distinct cities where the card has been utilized
- 'same_cano_separate_country_count': the number of transactions of a unique card in a certain country
- 'same_cano_separate_country_nunique': the number of distinct countries where the card has been utilized
#### 6. Card-related features
- 'number_of_unique_cards_owned': number of unique cards owned of a individual (chid)
- 'number_of_mcc_per_cano': number of unique mcc associated with a card (cano)
- 'mcc_frequency_per_cano': frequency of each mcc associated with a card (cano)

## Additional Notes
- Make sure to have the training.csv, public.csv, and private_1_processed.csv files in the same directory as preprocess.py.
- The module also provides detailed comments within the code for better understanding of each step.

