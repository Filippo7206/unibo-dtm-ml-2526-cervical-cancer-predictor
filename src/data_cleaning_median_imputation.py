import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

BASIC_CLEANED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\cleaned_data.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\data_after_imputation\\cleaned_data_median_imputation.csv'


# Setting up the imputation strategy for the missing values in the dataset.
def impute_missing_values(df):
    """
    Perform imputation of missing values for those columns still having some
    missing values after the initial cleaning steps.
    """
    num_imputer = SimpleImputer(missing_values=np.nan,strategy="median")
    int_missing_values = ["Number of sexual partners", "First sexual intercourse", "Num of pregnancies", 
                          "Hormonal Contraceptives (years)", "IUD (years)",
                          "STDs (number)", "STDs: Number of diagnosis"]
    
    df[int_missing_values] = num_imputer.fit_transform(df[int_missing_values])
    
    targets = ["Hinselmann", "Schiller", "Citology", "Biopsy"]
    exclude = int_missing_values + ["Age"] + targets

    cat_cols = [col for col in df.columns if col not in exclude]
    cat_imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
    
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    return df

if __name__ == "__main__":
    #executing the whole pipeline with the methods just defined
    print("Starting median-based imputation process...")

    data = pd.read_csv(BASIC_CLEANED_DATA_PATH)
    
    data = impute_missing_values(data)
    print("Missing values have been imputed.")

    data = data.drop_duplicates()
    print(f"After imputation, the current state of the data is: {data.shape}")

    data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")

