from __future__ import annotations

import argparse
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

    # Prevent duplicates: same location + same ide path
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest enDAQ .IDE files into measurement_run for a given site/pump folder."
    )
    p.add_argument("--site", required=True, help='Site name in DB, e.g. "Red"')
    p.add_argument("--pump", required=True, help='Pump name in DB, e.g. "P34A"')
    p.add_argument(
        "--folder",
        default=None,
        help=(
            "Folder containing .IDE files. "
            "If omitted, defaults to <data_root>/<site>/<pump>."
        ),
    )
    p.add_argument(
        "--operating-state",
        default="normal",
        help='Operating state to store, e.g. "normal", "loaded", etc.',
    )
    p.add_argument("--notes", default=None, help="Optional notes to attach to each reading.")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config()
    data_root = Path(cfg["data_root"])
    conn = get_connection(cfg["db_path"])

    site = args.site
    pump_name = args.pump

    # Determine where the IDEs live
    if args.folder:
        ide_dir = Path(args.folder)
        if not ide_dir.is_absolute():
            ide_dir = data_root / ide_dir
    else:
        ide_dir = data_root / site / pump_name

    if not ide_dir.exists():
        raise SystemExit(f"IDE folder does not exist: {ide_dir}")

    ide_files = sorted(ide_dir.glob("*.IDE"))
    if not ide_files:
        print(f"No .IDE files found in {ide_dir}")
        return

    print(f"Found {len(ide_files)} .IDE files in {ide_dir}")

    for ide_path in ide_files:
        # Example filename: 251113_155704_L01.IDE  -> loc = L01
        stem = ide_path.stem
        parts = stem.split("_")
        loc_name = parts[-1]  # "L01"

        location_id = get_location_id(conn, site, pump_name, loc_name)

        # Timestamp from file modification time (same as your current script)
        mtime = ide_path.stat().st_mtime
        timestamp = datetime.utcfromtimestamp(mtime).isoformat() + "Z"

        # Store relative path to data_root if possible
        try:
            ide_rel = str(ide_path.relative_to(data_root))
        except ValueError:
            # If user points outside data_root, store the absolute path as a fallback
            ide_rel = str(ide_path)

        reading_id = get_or_create_measurement_run(
            conn,
            location_id=location_id,
            timestamp_utc=timestamp,
            ide_file_rel=ide_rel,
            operating_state=args.operating_state,
            notes=args.notes,
        )

        print(f"{ide_rel}: {site}/{pump_name}/{loc_name} -> reading_id {reading_id}")


if __name__ == "__main__":
    main()
