### Data Preprocessing Pipeline ###

#Libraries
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler

# Load Data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# Clean Data Based on EDA
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop any non-informative columns
    df = df.drop(columns=["UDI", "Product ID"])

    #Drop leakage features
    df = df.drop(columns=["TWF", "HDF", "PWF", "OSF", "RNF"])
    return df

# Feature Encoding
def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(df, columns=["Type"], drop_first=True)
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove special characters from column names."""
    df.columns = (
        df.columns
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace(" ", "_")
    )
    return df

# Train-Test Split
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Machine_failure"])
    y = df["Machine_failure"]
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# Data Scaling
def scale_data(X_train, X_test):
    """Scale numerical features."""
    scaler = StandardScaler()

    numerical_cols = [
    "Air_temperature_K",
    "Process_temperature_K",
    "Rotational_speed_rpm",
    "Torque_Nm",
    "Tool_wear_min"
]

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, scaler


