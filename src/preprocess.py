import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

# Making height into inches
def parse_height(val):
    if pd.isna(val):
        return np.nan

    text = str(val).strip().lower()
    feet_match = re.search(r"(\d+)\s*ft", text)
    inch_match = re.search(r"(\d+)\s*in", text)

    feet = int(feet_match.group(1)) if feet_match else 0
    inches = int(inch_match.group(1)) if inch_match else 0


    if feet == 0 and inches == 0:
        return np.nan
    return feet * 12 + inches


def preprocess_splits(train_df, val_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()
    val_df = val_df.copy()

    #height in inches and hips missing indicator
    for df in [train_df, val_df, test_df]:
        df["height_in"] = df["height"].apply(parse_height)
        df["hips_is_missing"] = df["hips"].isna().astype(int)

    # Take the columns in num_cols from the train_df and put the medians of these columns inside the rows with missing data. 
    num_cols = ["size", "height_in", "hips", "bra size"]
    medians = {}
    for col in num_cols:
        medians[col] = train_df[col].median()
        train_df[col] = train_df[col].fillna(medians[col])
        val_df[col] = val_df[col].fillna(medians[col])
        test_df[col] = test_df[col].fillna(medians[col])

    # fill missing values in categorical columns with "unknown"
    cat_cols = ["category", "cup size"]
    for col in cat_cols:
        train_df[col] = train_df[col].fillna("unknown").astype(str)
        val_df[col] = val_df[col].fillna("unknown").astype(str)
        test_df[col] = test_df[col].fillna("unknown").astype(str)

    
    # One hot encode the categorical columns
    feature_cols = num_cols + ["hips_is_missing"] + cat_cols
    X_train = pd.get_dummies(train_df[feature_cols], columns=cat_cols, dtype=float)
    X_val = pd.get_dummies(val_df[feature_cols], columns=cat_cols, dtype=float)
    X_test = pd.get_dummies(test_df[feature_cols], columns=cat_cols, dtype=float)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0.0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)

    #Standardize numeric columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    y_train = train_df["fit_label"].values
    y_val = val_df["fit_label"].values
    y_test = test_df["fit_label"].values

    return X_train, y_train, X_val, y_val, X_test, y_test