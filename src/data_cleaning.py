import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from data_cleaning_knn_imputation import feature_scaling

RAW_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\raw.csv'
PROCESSED_DATA_PATH = 'C:\\unibo-dtm-ml-2526-cervical-cancer-predictor\\data\\cleaned_data.csv'

def load_and_clean_data(file_name = RAW_DATA_PATH):  
    """
    Initial cleaning steps based on EDA findings
    
    """
    #load the data and immediately proceed with "?" values conversion to NaN
    df = pd.read_csv(file_name, na_values='?')

    #remove the 23 duplicates found in EDA
    df.drop_duplicates(inplace=True)

    #conversion of the columns containing NaN values to numeric type
    df = df.apply(pd.to_numeric, errors="coerce").convert_dtypes()

    #dropping the two columns with more than 90% of missing values
    df = df.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])


    return df

def zero_variance_drop(df):
    """
    Drop the columns with zero variance, as they do not provide any useful information for the model.
    """

    #calculating the variance of each column and identifying those with zero variance
    variances = df.var()
    zero_var_cols = variances[variances == 0].index.tolist()

    #dropping the zero-variance columns and printing their names
    if zero_var_cols:
        df = df.drop(columns=zero_var_cols)
        for col in zero_var_cols:
            print(f"Column {col} has been dropped due to zero variance.")
    else:
        print("No zero-variance columns detected.")
    
    return df

def low_variance_aggr(df, threshold=0.01):
    """
    Aggregate low-variance features (all specific forms of STDs, as EDA unveiled)
    into two new columns: "STDs: Viral group" and "STDs: Bacterial group")
    """

    #lists of specific STDs to be aggregated into the two new columns, based on their nature (viral or bacterial)
    viral_group = ['STDs:genital herpes', 'STDs:Hepatitis B', 'STDs:HPV']
    bact_inf_group = ['STDs:pelvic inflammatory disease', 'STDs:molluscum contagiosum', 'STDs:vaginal condylomatosis']
    
    #creating the new aggregated columns by summing the values of the specific STD columns
    df['STDs: Viral group'] = df[viral_group].sum(axis=1)
    df['STDs: Bacterial group'] = df[bact_inf_group].sum(axis=1)

    #dropping the original specific STD columns after aggregation
    df = df.drop(columns=viral_group + bact_inf_group)

    return df

def corr_based_drop(df):
    """
    Drop features based on high correlation, as identified in EDA.
    """

    #dropping features with high correlation (corr >0.80) with other features, as identified in EDA
    high_corr_cols = ["STDs", "STDs:vulvo-perineal condylomatosis", "STDs: Number of diagnosis", "Dx:HPV"]
    df = df.drop(columns=high_corr_cols)

    #dropping other columns, characterized by less correlation (0.7 < corr < 0.80)
    #but still providing redundant information, as identified in EDA
    redundant_cols= ["Smokes", "Hormonal Contraceptives", "IUD"]
    df = df.drop(columns=redundant_cols)

    return df 


if __name__ == "__main__":
    #executing the whole pipeline with the methods just defined
    print("Starting data cleaning process...")

    #loading the raw data and performing the initial cleaning steps (removing duplicates, low-value columns, converting "?" to NaN)
    data = load_and_clean_data()
    print(f"Duplicates and low-value columns have been removed. The current state of the data is: {data.shape}")

    #next step: dropping columns with zero variance
    print("Proceeding with dropping zero-variance columns...")
    data = zero_variance_drop(data)
    print("All zero-variance columns have been dropped.")

    #next step: dropping features based on high correlation
    print("Proceeding with correlation-based feature dropping...")
    data = corr_based_drop(data)

    #next step: aggregating low-variance features into two new columns
    print("Proceeding with low-variance feature aggregation...")    
    data = low_variance_aggr(data)
    print("Low-variance features have been aggregated into new columns.")
    
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")