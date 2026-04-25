import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw categorical variables into numeric arrays."""
    data = df.copy()
    
    # Encode features
    if "GENDER" in data.columns:
        data["GENDER"] = data["GENDER"].map({"M": 1, "F": 0})
        
    # Encode target variable strictly for XGBoost
    if "LUNG_CANCER" in data.columns:
        data["LUNG_CANCER"] = data["LUNG_CANCER"].map({"NO": 0, "YES": 1})
        
    return data

def prepare_splits(data: pd.DataFrame, target_col: str = "LUNG_CANCER") -> Tuple:
    """Isolates targets and generates stratified training subsets."""
    features = data.drop(target_col, axis=1)
    target = data[target_col]
    
    return train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )