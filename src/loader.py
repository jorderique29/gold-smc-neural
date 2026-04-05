# src/loader.py
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).parent.parent


def load_mt5_csv(filename: str, base_dir: Path | None = None) -> pd.DataFrame:
    """
    Parse MT5 tab-separated export into a timezone-aware DataFrame.

    Columns returned: open, high, low, close, volume
    Index: 'timestamp' (UTC, datetime64[ns, UTC])
    Volume: uses TICKVOL column (VOL is always 0 from MT5 export).
    """
    path = (base_dir or PROJECT_DIR) / filename
    if not path.exists():
        path = PROJECT_DIR / "data" / "raw" / filename

    df = pd.read_csv(path, sep="\t")

    df["timestamp"] = pd.to_datetime(
        df["<DATE>"] + " " + df["<TIME>"],
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
    )
    df = df.set_index("timestamp")
    df.index.name = "timestamp"

    df = df.rename(columns={
        "<OPEN>": "open",
        "<HIGH>": "high",
        "<LOW>": "low",
        "<CLOSE>": "close",
        "<TICKVOL>": "volume",
    })[["open", "high", "low", "close", "volume"]]

    df = df.astype(float)
    df = df.sort_index()
    return df
