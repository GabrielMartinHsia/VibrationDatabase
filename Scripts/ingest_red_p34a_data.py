from pathlib import Path
from datetime import datetime

from vibtool.db import load_config, get_connection


def get_location_id(conn, site: str, pump_name: str, loc_name: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ml.location_id
        FROM measurement_location ml
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ? AND ml.name = ?
        """,
        (site, pump_name, loc_name),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"No location found for {site} / {pump_name} / {loc_name}")
    return row[0]


def get_or_create_measurement_run(
    conn,
    location_id: int,
    timestamp_utc: str,
    ide_file_rel: str,
    operating_state: str | None = None,
    sample_rate_hz: float | None = None,
    duration_s: float | None = None,
    notes: str | None = None,
) -> int:
    cur = conn.cursor()

    # Check if we already have a run for this location + file
    cur.execute(
        """
        SELECT reading_id
        FROM measurement_run
        WHERE location_id = ? AND ide_file_path = ?
        """,
        (location_id, ide_file_rel),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    cur.execute(
        """
        INSERT INTO measurement_run (
            location_id,
            timestamp_utc,
            operating_state,
            sample_rate_hz,
            duration_s,
            ide_file_path,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            location_id,
            timestamp_utc,
            operating_state,
            sample_rate_hz,
            duration_s,
            ide_file_rel,
            notes,
        ),
    )
    conn.commit()
    return cur.lastrowid


def main():
    cfg = load_config()
    data_root = Path(cfg["data_root"])
    conn = get_connection(cfg["db_path"])

    site = "Red"
    pump_name = "P34A"

    # Folder: Data/Red/P34A
    pump_dir = data_root / site / pump_name

    if not pump_dir.exists():
        raise SystemExit(f"Pump directory does not exist: {pump_dir}")

    # Look for all .IDE files in this folder
    ide_files = sorted(pump_dir.glob("*.IDE"))

    if not ide_files:
        print(f"No .IDE files found in {pump_dir}")
        return

    print(f"Found {len(ide_files)} .IDE files in {pump_dir}")


    for ide_path in ide_files:
        # Example filename: 251113_155704_L01.IDE
        stem = ide_path.stem                 # "251113_155704_L01"
        parts = stem.split("_")
        loc_code = parts[-1]                 # "L01"
        loc_name = loc_code                  # database uses L01, L02, ...

        # Get location_id from database
        location_id = get_location_id(conn, site, pump_name, loc_name)

        # Use file modification time for now
        mtime = ide_path.stat().st_mtime
        timestamp = datetime.utcfromtimestamp(mtime).isoformat() + "Z"

        # Store path relative to the data_root
        ide_rel = str(ide_path.relative_to(data_root))

        # Create or find a measurement_run record for this file
        reading_id = get_or_create_measurement_run(
            conn,
            location_id=location_id,
            timestamp_utc=timestamp,
            ide_file_rel=ide_rel,
            operating_state="normal",
        )

        print(f"{ide_rel}: location {loc_name} (id={location_id}) → reading_id {reading_id}")



if __name__ == "__main__":
    main()
