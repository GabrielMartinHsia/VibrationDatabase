from pathlib import Path
import sqlite3
from vibtool.db import load_config

def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("Renaming run_id to reading_id in measurement_run...")
    cur.execute("ALTER TABLE measurement_run RENAME COLUMN run_id TO reading_id;")

    print("Rebuilding fft_spectrum with new schema...")
    cur.execute("ALTER TABLE fft_spectrum RENAME TO fft_old;")

    cur.execute("""
        CREATE TABLE fft_spectrum (
            reading_id      INTEGER NOT NULL,
            axis            TEXT NOT NULL CHECK (axis IN ('X','Y','Z')),
            nfft            INTEGER NOT NULL,
            sample_rate_hz  REAL NOT NULL,
            window          TEXT,
            spectrum_type   TEXT NOT NULL,
            spectrum_blob   BLOB NOT NULL,
            PRIMARY KEY (reading_id, axis),
            FOREIGN KEY (reading_id) REFERENCES measurement_run(reading_id)
        );
    """)

    print("Copying FFT rows...")
    cur.execute("""
        INSERT INTO fft_spectrum (
            reading_id, axis, nfft, sample_rate_hz, window, spectrum_type, spectrum_blob
        )
        SELECT
            run_id, axis, nfft, sample_rate_hz, window, spectrum_type, spectrum_blob
        FROM fft_old;
    """)

    print("Dropping old fft table...")
    cur.execute("DROP TABLE fft_old;")

    conn.commit()
    conn.close()
    print("Migration complete!")

if __name__ == "__main__":
    main()
