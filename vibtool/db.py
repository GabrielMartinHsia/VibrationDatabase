import sqlite3
from pathlib import Path
import yaml
from .schema import SCHEMA_SQL


def load_config(config_path: str | Path = "config.yaml") -> dict:
    """Load settings from config.yaml."""
    config_path = Path(config_path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_connection(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection to the SQLite database.

    If the folder doesn't exist yet, create it.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    # Foreign keys are off by default in SQLite connections
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create all tables/indexes (if they don't already exist)."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cur.fetchall())


def ensure_measurement_type_field(conn):
    cur = conn.cursor()
    # Add column if missing
    cur.execute("PRAGMA table_info(measurement_run)")
    cols = {row[1] for row in cur.fetchall()}
    if "measurement_type" not in cols:
        cur.execute("ALTER TABLE measurement_run ADD COLUMN measurement_type TEXT DEFAULT 'route'")
        conn.commit()


def ensure_ide_fingerprint_fields(conn: sqlite3.Connection) -> None:
    """Best-effort migration for older DBs.

    Adds stable file identity fields on measurement_run if missing:
      - ide_sha256
      - device_serial
      - recording_start_utc

    Also creates helpful indexes.

    Safe to call repeatedly.
    """
    # measurement_run must exist (initialize_schema should be called first)
    if not _has_column(conn, "measurement_run", "ide_sha256"):
        conn.execute("ALTER TABLE measurement_run ADD COLUMN ide_sha256 TEXT;")
    if not _has_column(conn, "measurement_run", "device_serial"):
        conn.execute("ALTER TABLE measurement_run ADD COLUMN device_serial TEXT;")
    if not _has_column(conn, "measurement_run", "recording_start_utc"):
        conn.execute("ALTER TABLE measurement_run ADD COLUMN recording_start_utc TEXT;")

    # Indexes (SQLite supports IF NOT EXISTS)
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_measurement_run_ide_sha256
        ON measurement_run(ide_sha256)
        WHERE ide_sha256 IS NOT NULL;
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_measurement_run_location_time
        ON measurement_run(location_id, timestamp_utc);
        """
    )

    conn.commit()
