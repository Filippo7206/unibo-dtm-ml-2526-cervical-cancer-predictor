import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os 

RAW_DATA_PATH = os.path.join(os.getcwd(), 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(os.getcwd(), 'data', 'processed')

def load_clean_data(file_name = RAW_DATA_PATH): 
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