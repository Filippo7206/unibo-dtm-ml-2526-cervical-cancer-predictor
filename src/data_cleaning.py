import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os 

RAW_DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')

def load_and_clean_data(file_name = RAW_DATA_PATH): 
    """
    Initial cleaning steps based on EDA findings
    
    """
    #load the data and immediately proceed with "?" vlaues conversion to NaN
    df = pd.read_csv(file_name, na_values='?')

    #remove the 23 duplicates found in EDA
    df.drop_duplicates(inplace=True)

    #drop the two columns having too many missing values
    df.drop(columns=["STDs: Time since first diagnosis","STDs: Time since last diagnosis"], inplace=True)

    #conversion of the columns containing NaN values to numeric type
    df = df.apply(pd.to_numeric, errors="coerce").convert_dtypes()

    return df

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
    print("Starting data cleaning process...")

    #loading the raw data and performing the initial cleaning steps (removing duplicates, low-value columns, converting "?" to NaN)
    data = load_and_clean_data()
    print(f"Duplicates and low-value columns have been removed. The current state of the data is: {data.shape}")

    #next step: imputation of missing values for the columns still having some missing values after the initial cleaning steps.
    print("Proceeding with imputation of missing values...")
    data = impute_missing_values(data)
    print("Missing values have been imputed.")

    #save the cleaned data for modeling 
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    output_path = os.path.join(PROCESSED_DATA_DIR, "processed.csv")
    data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

