# unibo-dtm-ml-2526-cervical-cancer-predictor

## Project Overview
This ML project revolves around the prediction of indicators/diagnosis of cervical cancer.   

## Dataset Description
The dataset to be used for the project was taken from the UC Irvine Machine Learning Repository, and it was originally collected at 'Hospital Universitario de Caracas' in Caracas, Venezuela. It comprises demographic information, habits, and historic medical records of 858 patients. Additionally, the results of the 4 diagnostic tests related to the matter (Hinselmann, Schiller, Citology, Biopsy) are featured for each sample in the dataset. Missing values do exist, as several patients decided not to answer some of the questions because of privacy concerns. 

## ML Task Definition
The specific goal of the project is to make the ML model predict the results of the aforementioned 4 diagnostic tests and, in order to do this, a multi-label binary classification task will be carried out through the main ML techniques: logistic regression (one vs rest), random forest, XGBoost, SVM, multi-label k-NN, and multilabel NNs eventually (to be included only if results are satisfying).

## Pipeline Summary
This project utilizes a custom, mathematically rigorous preprocessing and modeling pipeline designed for a highly imbalanced multi-target clinical dataset.

- Exploratory Data Analysis (EDA): Deep-dive visual analysis of clinical risk factors, missingness, and target correlations (notebooks/01_eda.ipynb, 02_eda.ipynb).

- Data Cleaning & Scaling: Consolidation of sparse medical attributes, handling of skewed distributions, and MinMaxScaler application (src/data_cleaning.py).

- A/B Imputation Testing: Implementation of two parallel imputation strategies to handle missing clinical data: a baseline median/frequency approach (src/median_and_freq_imputation.py) and a localized spatial approach (src/knn_imputation.py).

- Multi-Target Classification: Simultaneous prediction of four clinical outcomes (Biopsy, Hinselmann, Schiller, Citology) evaluated in dedicated modeling environments (notebooks/knn_modeling.ipynb, median_modeling.ipynb).

- Robust Cross-Validation: Utilization of a custom multi-label stratifier (src/utils/ml_stratifiers.py) to ensure fair minority class distribution across all training folds.

## Repository Structure
```text
├── assets/                            # Plots, countplots, and correlation matrices
├── data/
│   ├── raw.csv                        # Original unprocessed dataset
│   ├── cleaned_data.csv               # Dataset after initial cleaning
│   └── data_after_imputation/         
│       ├── knn_imputed.csv            # Final dataset used for modeling (Champion)
│       └── median_and_freq_imputed.csv # Baseline dataset used for A/B testing
├── notebooks/                         
│   ├── 01_eda.ipynb                   # Initial Exploratory Data Analysis
│   ├── 02_eda.ipynb                   # Advanced EDA and Feature Target Analysis
│   ├── knn_modeling.ipynb             # Model training/eval on KNN-imputed data
│   └── median_modeling.ipynb          # Model training/eval on baseline data
├── report/                            
│   ├── ML_CC_presentation.pdf         # Slide deck for project presentation
│   └── Multilabel_Classification_Cervical_Cancer_Diagnosis.pdf # Final academic report
├── src/                               
│   ├── data_cleaning.py               # Feature dropping and initial parsing
│   ├── knn_imputation.py              # KNN Imputation and Isolation Forest anomaly detection
│   ├── median_and_freq_imputation.py  # Baseline median/mode imputation
│   └── utils/
│       └── ml_stratifiers.py          # Custom Stratified K-Fold for multi-label data
├── .gitignore
├── LICENSE
├── README.md                          
└── requirements.txt                   # Project dependencies
```

## Results 
The primary clinical objective was to achieve a Biopsy Precision > 0.20 while maintaining a safe Recall > 0.60 to minimize false negatives in cancer detection.

The project demonstrated Algorithmic Synergy: Preparing the data with a distance-based imputer (KNN) preserved spatial clinical variance, creating a highly optimized data map that allowed the classifiers to drastically outperform the median-imputed baseline.

Top Performers (Biopsy Target on KNN-Imputed Data):

- KNN Classifier: Precision 0.270 | Recall 0.655 | F1-Score 0.325 (Exceeded all clinical targets)

- XGBoost: Precision 0.200 | Recall 0.655 | F1-Score 0.288 (Met clinical targets)

Note: The median_modeling.ipynb experiments proved that standard median/frequency imputation collapsed the feature space, causing all tested classifiers (including tree-based and linear models) to fail the 0.20 precision threshold.

## Limitations & Future Work
Limitations:

- Extreme Class Imbalance: The inherent rarity of positive outcomes (especially for Hinselmann and Citology) limits the boundary-finding capabilities of standard linear classifiers.

- Global Imputation Risk: Performing static imputation prior to cross-validation introduces a negligible but theoretically present risk of data leakage.

Future Work:

- Hyperparameter Optimization: Implement comprehensive GridSearchCV on the champion KNN classifier to fine-tune neighbors (k), distance metrics, and weight distributions.

- Advanced Minority Sampling: Explore synthetic data generation techniques (e.g., SMOTE, ADASYN) within the cross-validation loops to provide models with better minority-class representation without overfitting.

- Feature Engineering: Extract non-linear combinations of tracked risk factors (e.g., compounding Age + Smokes (years)) to further assist linear and tree-based classification boundaries.
