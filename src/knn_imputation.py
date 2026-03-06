import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest


BASIC_CLEANED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\cleaned_data.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\data_after_imputation\\knn_imputed.csv'

targets = ["Hinselmann", "Schiller", "Citology", "Biopsy"]    

def feature_scaling(df): 
    """
    Perform feature scaling before proceeding with KNN imputation,
    so as to avoid the distance calculation 
    being biased by the different scales of the features.
    """
    features = [col for col in df.columns if col not in targets]

    scaler = MinMaxScaler() 
    scaled_data = scaler.fit_transform(df[features])
    scaled_df = df.copy()
    scaled_df[features] = scaled_data

    #returning the scaler object so to then eventually perform the inverse transformation
    return scaled_df, scaler

def KNN_imputing(df, n_neighbors=29):
    """
    Perform KNN imputation of the missing values 
    for those columns still having some
    k-value is set to 29 for the moment, as it is the square root 
    of the number of samples in the dataset, 
    but it will be further tuned in the next steps of the project.
    """
    imputer = KNNImputer(missing_values=np.nan, n_neighbors=n_neighbors,
                         weights="distance")
    
    
    features = [col for col in df.columns if col not in targets]

    df[features] = imputer.fit_transform(df[features])
    
    return df

def isolation_forest_anomaly_detection(df):
    """
    Perform anomaly detection using Isolation Forest on the imputed dataset.
    This method identifies anomalies by isolating observations in the feature space.
    """
    
    features = [col for col in df.columns if col not in targets]
    clf = IsolationForest(n_estimators=100, contamination='auto', random_state=42)

    df['anomaly_label'] = clf.fit_predict(df[features])
    df['anomaly_score'] = clf.decision_function(df[features])

    return df

def restore_clinical_units(df, scaler):

    #defining the original features to be re-scaled (excluding anomaly_label and anomaly_score)
    original_features = list(scaler.feature_names_in_)

    #reverse the scale of the features after all operations have been performed
    df[original_features] = scaler.inverse_transform(df[original_features])

    #doing the same with the previously log-transformed features, so not to distort the medical reality of the data
    cols_to_transform = ["Number of sexual partners", "Smokes (years)", 
                     "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)"]

    df[cols_to_transform] = np.expm1(df[cols_to_transform])

    #separate the continuous columns from the discrete ones
    continuous_cols = ["Smokes (years)", "Smokes (packs/year)", 
        "Hormonal Contraceptives (years)", "IUD (years)"]
    discrete_cols = [col for col in original_features if col not in continuous_cols]

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
    imputed_data = KNN_imputing(scaled_data)
    print("Missing values have been imputed.")
    
    #performing anomaly detection on the imputed dataset
    analyzed_data = isolation_forest_anomaly_detection(imputed_data)
    print("Anomaly detection completed.")

    #restore the data from scaling and log transform (data un-squashing)
    final_data = restore_clinical_units(analyzed_data, scaler)

    final_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")


