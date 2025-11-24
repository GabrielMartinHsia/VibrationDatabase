from pathlib import Path
import sqlite3
from vibtool.db import load_config


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Checking for band_level table...")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='band_level';")
    if not cur.fetchone():
        print("No band_level table found. Nothing to migrate.")
        return

    # Check if migration already done
    cur.execute("PRAGMA table_info(band_level);")
    cols = [row[1] for row in cur.fetchall()]
    if "reading_id" in cols:
        print("band_level already migrated. Nothing to do.")
        return

    print("Migrating band_level table...")

    # 1. Rename old table
    cur.execute("ALTER TABLE band_level RENAME TO band_old;")

    # 2. Create new band_level table with reading_id
    cur.execute("""
        CREATE TABLE band_level (
            reading_id   INTEGER NOT NULL,
            axis         TEXT NOT NULL CHECK (axis IN ('X','Y','Z')),
            f_low_hz     REAL NOT NULL,
            f_high_hz    REAL NOT NULL,
            metric       TEXT NOT NULL,
            value        REAL NOT NULL,
            PRIMARY KEY (reading_id, axis, f_low_hz, f_high_hz, metric),
            FOREIGN KEY (reading_id) REFERENCES measurement_run(reading_id)
        );
    """)

    # 3. Copy data from old table
    print("Copying rows from band_old → band_level...")
    cur.execute("""
        INSERT INTO band_level (
            reading_id, axis, f_low_hz, f_high_hz, metric, value
        )
        SELECT
            run_id, axis, f_low_hz, f_high_hz, metric, value
        FROM band_old;
    """)

    # 4. Drop old table
    print("Dropping band_old...")
    cur.execute("DROP TABLE band_old;")

    conn.commit()
    conn.close()
    print("band_level migration complete!")


if __name__ == "__main__":
    main()
