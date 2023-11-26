# E.Sun - Credit Card Fraud Detection

Submission for AI CUP 2023 E.Sun Artificial Intelligence Open Challenge - Credit Card Fraud Detection

## IMPORTANT

Please put the datasets ('training.csv', 'public_processed.csv', 'public.csv', and 'private_1_processed.csv') into the ```data/``` directory.

## Method Overview

1. Data Preprocessing: Cleaning and preprocessing the provided data to prepare it for model training.
2. Feature Engineering: Creating new features that enhance the models' ability to detect fraudulent activities.
3. Model Development: Building machine learning models (e.g., XGBoost, LightGBM, CatBoost) to effectively classify transactions as genuine or fraudulent.
4. Hyperparameter Tuning: Fine-tuning the models to achieve optimal performance.
5. Model Evaluation: Assessing model performance using F1-Score.
6. Ensemble Method: Combine the results from each model using max voting to make further generalization, preventing overfitting and improving further prediction.

## File Content

- ```Model/```
	- ```model.py```: Stores code for hyparameter tuning by GridSearchCV, and code for training and predicting using the models.
- ```Preprocess/```
	- ```preprocess.py```: Stores code for data cleaning and preprocessing. Also contains code for feature engineering.
- ```requirements.txt```: Required packages for the code.
- ```main.py```: Execute this file to perform the entire complete process.

 ## Implementation Process Example

 ```
# Install required packages
$ pip install -r requirements.txt 

# Run code
$ python main.py
```
Use ```python -W "ignore" main.py``` to ignore warnings
