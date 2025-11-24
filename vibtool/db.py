import sqlite3
from pathlib import Path
import yaml
from .schema import SCHEMA_SQL


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """
    Load settings from config.yaml.
    """
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.
    If the folder doesn't exist yet, create it.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    return conn


def initialize_schema(conn):
    """
    Create all tables (if they don't already exist).
    """
    conn.executescript(SCHEMA_SQL)
    conn.commit()

