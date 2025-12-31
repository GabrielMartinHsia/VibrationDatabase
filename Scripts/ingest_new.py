from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path

from vibtool.db import load_config, get_connection, initialize_schema, ensure_ide_fingerprint_fields


_LOC_RE = re.compile(r"L\d{2}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Bulk ingest any new .IDE files under the data root. "
            "Infers site and machine from folder structure: <data_root>/<site>/<machine>/..."
        )
    )
    p.add_argument("--root", default=None, help="Override data_root (otherwise uses config).")
    p.add_argument("--site", default=None, help="Optional: ingest only this site folder (e.g. Red).")
    p.add_argument(
        "--machine",
        default=None,
        help="Optional: ingest only this machine folder (e.g. P34A). (Requires --site)",
    )
    p.add_argument("--dry-run", action="store_true", help="Print what would be ingested, but do not write DB.")
    p.add_argument(
        "--operating-state",
        default="normal",
        help='Operating state to store, e.g. "normal", "loaded", etc.',
    )
    p.add_argument("--notes", default=None, help="Optional notes to attach to each ingested reading.")
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def infer_site_machine(data_root: Path, ide_path: Path) -> tuple[str, str]:
    """Expect <data_root>/<site>/<machine>/.../<file>.IDE."""
    rel = ide_path.relative_to(data_root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"IDE path not deep enough to infer site/machine: {rel}")
    return parts[0], parts[1]


def infer_location_name(ide_path: Path) -> str:
    """Supports both:
      - YYMMDD_HHMMSS_L01.IDE (legacy)
      - YYMMDD_R01_L01.IDE    (recommended)

    We only require the final token to be L##.
    """
    stem = ide_path.stem
    parts = stem.split("_")
    if not parts:
        raise ValueError(f"Could not parse filename: {ide_path.name}")

    candidate = parts[-1]
    if not _LOC_RE.match(candidate):
        raise ValueError(f"Filename does not end with location token like L01: {ide_path.name}")

    # Normalize to 'L01' style
    candidate = candidate.upper()
    return candidate


def get_location_id(conn, site: str, machine: str, loc_name: str) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ml.location_id
        FROM measurement_location ml
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ? AND ml.name = ?
        """,
        (site, machine, loc_name),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"No location found for {site}/{machine}/{loc_name}")
    return row[0]


def find_existing_by_hash(conn, ide_sha256: str) -> int | None:
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT reading_id FROM measurement_run WHERE ide_sha256 = ? LIMIT 1",
            (ide_sha256,),
        )
    except Exception:
        # Column may not exist if migrations weren't run, but we run them, so this is just belt+suspenders
        return None
    row = cur.fetchone()
    return int(row[0]) if row else None


def find_existing_by_path(conn, ide_path: str) -> tuple[int, str | None] | None:
    """Returns (reading_id, ide_sha256) if an entry exists for ide_file_path."""
    cur = conn.cursor()
    cur.execute(
        "SELECT reading_id, ide_sha256 FROM measurement_run WHERE ide_file_path = ? LIMIT 1",
        (ide_path,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return int(row[0]), (row[1] if row[1] else None)


def insert_measurement_run(
    conn,
    location_id: int,
    timestamp_utc: str,
    ide_file_path: str,
    ide_sha256: str,
    operating_state: str,
    notes: str | None,
    measurement_type: str,
    dry_run: bool,
) -> int | None:
    if dry_run:
        return None

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO measurement_run (
            location_id,
            timestamp_utc,
            operating_state,
            ide_file_path,
            ide_sha256,
            notes,
            measurement_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (location_id, timestamp_utc, operating_state, ide_file_path, ide_sha256, notes, measurement_type),
    )
    conn.commit()
    return cur.lastrowid


def update_existing_hash(conn, reading_id: int, ide_sha256: str, dry_run: bool) -> None:
    if dry_run:
        return
    conn.execute(
        "UPDATE measurement_run SET ide_sha256 = ? WHERE reading_id = ?",
        (ide_sha256, reading_id),
    )
    conn.commit()


def main() -> None:
    args = parse_args()

    cfg = load_config()
    data_root = Path(args.root) if args.root else Path(cfg["data_root"])

    conn = get_connection(cfg["db_path"])
    initialize_schema(conn)
    ensure_ide_fingerprint_fields(conn)

    # Choose search root based on optional filters
    search_root = data_root
    if args.site and args.machine:
        search_root = data_root / args.site / args.machine
    elif args.site:
        search_root = data_root / args.site

    if args.machine and not args.site:
        print("Note: --machine without --site is ambiguous in the folder tree; ignoring --machine filter.")

    if not search_root.exists():
        raise SystemExit(f"Search root does not exist: {search_root}")

    ide_files = sorted(search_root.rglob("*.IDE"))

    ingested = 0
    skipped = 0
    updated = 0
    errors = 0

    for ide_path in ide_files:
        try:
            site, machine = infer_site_machine(data_root, ide_path)
        except Exception:
            # Not in the expected folder structure
            continue

        if args.site and site != args.site:
            continue
        if args.site and args.machine and machine != args.machine:
            continue

        # Relative path for portability
        try:
            ide_rel = str(ide_path.relative_to(data_root))
        except ValueError:
            ide_rel = str(ide_path)

        parts_lower = [p.lower() for p in ide_path.parts]
        measurement_type = "sweep" if "sweeps" in parts_lower else "route"

        try:
            loc_name = infer_location_name(ide_path)
        except Exception as e:
            print(f"SKIP (bad name): {ide_rel} ({e})")
            errors += 1
            continue

        # Compute stable fingerprint
        try:
            ide_hash = sha256_file(ide_path)
        except Exception as e:
            print(f"SKIP (hash failed): {ide_rel} ({e})")
            errors += 1
            continue

        # First: de-dup by hash
        existing_id = find_existing_by_hash(conn, ide_hash)
        if existing_id is not None:
            skipped += 1
            continue

        # Second: de-dup by path (helps during transition/backfill)
        existing_path_row = find_existing_by_path(conn, ide_rel)
        if existing_path_row is not None:
            existing_id2, existing_hash2 = existing_path_row
            if existing_hash2 is None:
                update_existing_hash(conn, existing_id2, ide_hash, args.dry_run)
                updated += 1
                print(f"UPDATED HASH: {ide_rel} -> reading_id={existing_id2}")
            skipped += 1
            continue

        # Look up location_id (requires machine+location rows already exist in DB)
        try:
            location_id = get_location_id(conn, site, machine, loc_name)
        except Exception as e:
            print(f"SKIP (unknown location): {ide_rel} ({e})")
            errors += 1
            continue

        # Use file modification time as a reasonable default ordering timestamp
        mtime = ide_path.stat().st_mtime
        timestamp = datetime.utcfromtimestamp(mtime).isoformat() + "Z"

        reading_id = insert_measurement_run(
            conn,
            location_id=location_id,
            timestamp_utc=timestamp,
            ide_file_path=ide_rel,
            ide_sha256=ide_hash,
            operating_state=args.operating_state,
            notes=args.notes,
            measurement_type=measurement_type,
            dry_run=args.dry_run,
        )

        print(
            f"{'WOULD INGEST' if args.dry_run else 'INGESTED'}: {ide_rel} -> {site}/{machine}/{loc_name}"
            + (f" reading_id={reading_id}" if reading_id else "")
        )
        ingested += 1

    print(
        f"\nDone. New ingested: {ingested}. Skipped (already present): {skipped}. "
        f"Updated hashes: {updated}. Filename/errors skipped: {errors}."
    )


if __name__ == "__main__":
    main()
