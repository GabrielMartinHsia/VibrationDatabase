from __future__ import annotations

import hashlib
from pathlib import Path

from vibtool.db import load_config, get_connection, initialize_schema, ensure_ide_fingerprint_fields


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg = load_config()
    data_root = Path(cfg["data_root"])
    conn = get_connection(cfg["db_path"])

    initialize_schema(conn)
    ensure_ide_fingerprint_fields(conn)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT reading_id, ide_file_path
        FROM measurement_run
        WHERE ide_sha256 IS NULL OR ide_sha256 = ''
        """
    )

    rows = cur.fetchall()
    if not rows:
        print("No rows missing ide_sha256. Nothing to do.")
        return

    updated = 0
    missing_files = 0

    for reading_id, ide_file_path in rows:
        ide_path = Path(ide_file_path)
        if not ide_path.is_absolute():
            ide_path = data_root / ide_path

        if not ide_path.exists():
            print(f"MISSING FILE: reading_id={reading_id} path={ide_file_path}")
            missing_files += 1
            continue

        try:
            digest = sha256_file(ide_path)
        except Exception as e:
            print(f"HASH FAILED: reading_id={reading_id} path={ide_file_path} ({e})")
            continue

        conn.execute(
            "UPDATE measurement_run SET ide_sha256 = ? WHERE reading_id = ?",
            (digest, reading_id),
        )
        updated += 1

    conn.commit()
    print(f"Backfill complete. Updated: {updated}. Missing files: {missing_files}.")


if __name__ == "__main__":
    main()
