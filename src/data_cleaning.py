import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from data_cleaning_knn_imputation import feature_scaling
import os 

RAW_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\raw.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\cleaned_data.csv'

def load_and_clean_data(file_name = RAW_DATA_PATH):  
    """
    Initial cleaning steps based on EDA findings
    
    """
    #load the data and immediately proceed with "?" vlaues conversion to NaN
    df = pd.read_csv(file_name, na_values='?')

    #remove the 23 duplicates found in EDA
    df.drop_duplicates(inplace=True)

    #conversion of the columns containing NaN values to numeric type
    df = df.apply(pd.to_numeric, errors="coerce").convert_dtypes()

    return df

def zero_variance_drop(df):
    """
    Drop the columns with zero variance, as they do not provide any useful information for the model.
    """

    #scaling features before calculating the variance    
    scaled_data, scaler = feature_scaling(df)

    scaled_df = df.copy()
    scaled_df[scaled_data.columns] = scaled_data

    #calculating the variance of each feature and identifying those with zero variance
    variances = scaled_data.var()
    zero_var_cols = variances[variances == 0].index.tolist()

    #dropping the zero-variance columns and printing their names
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        for col in zero_var_cols:
            print(f"Column {col} has been dropped due to zero variance.")
    else:
        print("No zero-variance columns detected.")
    
    return df