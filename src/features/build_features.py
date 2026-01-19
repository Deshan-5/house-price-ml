"""
Feature engineering logic:
- Encode categorical features
- Scale numerical features
- Save model-ready data
"""

from pathlib import Path
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------
# Load config
# ---------------------------
def load_config(config_path: Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------
# Load processed data
# ---------------------------
def load_processed_data(processed_path: Path) -> pd.DataFrame:
    file_path = processed_path / "train_processed.csv"
    df = pd.read_csv(file_path)
    logger.info(f"Loaded processed data with shape: {df.shape}")
    return df


# ---------------------------
# Feature engineering
# ---------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Separate target
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # Identify column types
    num_cols = X.select_dtypes(exclude="object").columns
    cat_cols = X.select_dtypes(include="object").columns

    logger.info(f"Numerical columns: {list(num_cols)}")
    logger.info(f"Categorical columns: {list(cat_cols)}")

    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X[num_cols])
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=num_cols)

    # Encode categorical features
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    X_cat_encoded = encoder.fit_transform(X[cat_cols])
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    X_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names)

    # Combine features
    X_final = pd.concat([X_num_scaled, X_cat_encoded], axis=1)

    # Reattach target
    X_final["SalePrice"] = y.values

    logger.info(f"Final feature matrix shape: {X_final.shape}")
    return X_final


# ---------------------------
# Save features
# ---------------------------
def save_features(df: pd.DataFrame, output_path: Path):
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / "train_features.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Saved feature data to: {file_path}")


# ---------------------------
# Pipeline
# ---------------------------
def feature_pipeline(config_path: Path):
    config = load_config(config_path)

    processed_path = PROJECT_ROOT / config["data"]["processed_path"]

    df_processed = load_processed_data(processed_path)
    df_features = build_features(df_processed)
    save_features(df_features, processed_path)


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    feature_pipeline(config_path)
