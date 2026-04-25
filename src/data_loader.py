import pandas as pd
from pathlib import Path

def load_data(file_path: str | Path) -> pd.DataFrame:
    """Imports dataset from local storage."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Target data not found at: {path}")
    
    return pd.read_csv(path)