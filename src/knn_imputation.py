import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer


BASIC_CLEANED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\cleaned_data.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\data_after_imputation\\cleaned_data_knn_imputation.csv'

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

def KNN_imputing(df,scaler, n_neighbors=29):
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
    print("Starting KNN imputation process...")

    data = pd.read_csv(BASIC_CLEANED_DATA_PATH)

    #scaling the features before KNN imputation
    scaled_data, scaler = feature_scaling(data)

    #KNN imputation of the missing values 
    imputed_data = KNN_imputing(scaled_data,scaler)
    print("Missing values have been imputed.")

    final_data = imputed_data.drop_duplicates()
    print(f"After imputation, the current state of the data is: {final_data.shape}")

    
    final_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")


