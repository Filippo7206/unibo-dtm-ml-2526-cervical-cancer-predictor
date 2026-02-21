import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os 
from data_cleaning_median_imputation import load_and_clean_data
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

RAW_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\raw.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\processed_data\\cleaned_data_knn_imputation.csv'

def feature_scaling(df): 
    """
    Perform feature scaling before proceeding with KNN imputation,
    so as to avoid the distance calculation 
    being biased by the different scales of the features.
    """
    targets = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    features = [col for col in df.columns if col not in targets]
    
    scaler = MinMaxScaler() 
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = df.copy()
    scaled_df[features] = scaled_data

    #returning the scaler object so to then eventually perform the inverse transformation
    return scaled_df, scaler

def KNN_imputation(df,scaler, n_neighbors=29):
    """
    Perform KNN imputation of the missing values 
    for those columns still having some
    k-value is set to 29 for the moment, as it is the square root 
    of the number of samples in the dataset, 
    but it will be further tuned in the next steps of the project.
    """
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                         weights="distance")
    
    
    targets = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    features = [col for col in df.columns if col not in targets]

    df[features] = imputer.fit_transform(df[features])

    #bringing back integer-dtype features to their original scale before rounding them to the nearest integer
    df[features] = scaler.inverse_transform(df[features])

    #before rounding the scaled integer values, we need to separate the continuous columns from such discrete ones
    continuous_cols = ["Smokes (years)", "Smokes (packs/year)", 
        "Hormonal Contraceptives (years)", "IUD (years)"]
    discrete_cols = [col for col in features if col not in continuous_cols]

    #rounding the discrete features to the nearest integer
    for col in discrete_cols: 
        df[col] = df[col].round().astype(int)
    
    return df

if __name__ == "__main__":
    
    #executing the whole pipeline with the methods just defined
    print("Starting data cleaning process...")

    #loading the raw data and performing the initial cleaning steps (removing duplicates, low-value columns, converting "?" to NaN)
    data = load_and_clean_data()
    print(f"Duplicates and low-value columns have been removed. The current state of the data is: {data.shape}")

    #scaling the features before KNN imputation
    scaled_data, scaler = feature_scaling(data)

    #KNN imputation of the missing values 
    print("Proceeding with KNN imputation of missing values...")
    imputed_data = KNN_imputation(scaled_data,scaler)
    print("Missing values have been imputed.")

    final_data = imputed_data.drop_duplicates()
    print(f"After imputation, the current state of the data is: {final_data.shape}")

    
    final_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")


