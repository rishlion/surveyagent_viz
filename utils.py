import pandas as pd
from pathlib import Path

def load_transcripts(file_path: Path) -> pd.DataFrame:
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)
    elif file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file type")
