
from src.utils.logger import get_logger
logger = get_logger(__name__)

"""
Data cleaning and preprocessing logic.
"""
from pathlib import Path
print("FILE:", Path(__file__).resolve())
print("PARENTS:", list(Path(__file__).resolve().parents))

def preprocess_data(df):
    """
    Clean raw data and return processed dataframe.
    """
    pass
"""
Data cleaning and preprocessing logic.
"""
from pathlib import Path
import pandas as pd
import yaml

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]





def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    df_raw = load_raw_data(raw_path)
    logger.info(f"Loaded raw data with shape: {df_raw.shape}")

    train_file = Path(raw_path) / "train.csv"
    return pd.read_csv(train_file)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing:
    - Separate target
    - Handle missing values
    """
    df = df.copy()

    # Drop Id column (identifier, not a feature)
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
        logger.info(f"Dropped 'Id' column.")

    # Separate target
    target = df["SalePrice"]
    df.drop(columns=["SalePrice"], inplace=True)

    # Numerical & categorical columns
    num_cols = df.select_dtypes(exclude="object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    logger.info(f"Filled missing numerical values with median for columns: {list(num_cols)}")

    # categorical columns 
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna("Missing")
    logger.info(f"Filled missing categorical values with 'Missing' for columns: {list(cat_cols)}")
    # Reattach target
    df["SalePrice"] = target

    return df


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to disk."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path / "train_processed.csv", index=False)
    logger.info(f"Saved processed data to {output_path / 'train_processed.csv'}")       


if __name__ == "__main__":
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(config_path)

    raw_path = PROJECT_ROOT / config["data"]["raw_path"]
    processed_path = PROJECT_ROOT / config["data"]["processed_path"]

    df_raw = load_raw_data(raw_path)
    df_processed = preprocess_data(df_raw)

    save_processed_data(df_processed, processed_path)

