from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from vibtool.db import load_config, get_connection


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(
        description="Sync measurement_run.ide_file_path to current disk paths by matching ide_sha256."
    )
    p.add_argument("--dry-run", action="store_true", help="Show changes without writing DB.")
    p.add_argument("--root", default=None, help="Override data_root from config.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    data_root = Path(args.root) if args.root else Path(cfg["data_root"])
    conn = get_connection(cfg["db_path"])

    # 1) Load DB hashes
    cur = conn.cursor()
    cur.execute("SELECT reading_id, ide_sha256, ide_file_path FROM measurement_run")
    rows = cur.fetchall()

    db_by_hash = {}
    missing_hash = 0
    for reading_id, ide_sha256, ide_file_path in rows:
        if not ide_sha256:
            missing_hash += 1
            continue
        db_by_hash.setdefault(ide_sha256, []).append((reading_id, ide_file_path))

    if missing_hash:
        print(f"Note: {missing_hash} rows have no ide_sha256. Run: python -m Scripts.backfill_ide_sha256")

    # 2) Scan disk and build hash -> relpath mapping
    disk_by_hash: dict[str, list[str]] = {}
    ide_files = list(data_root.rglob("*.IDE"))

    for p in ide_files:
        try:
            rel = str(p.relative_to(data_root))
        except ValueError:
            continue
        h = sha256_file(p)
        disk_by_hash.setdefault(h, []).append(rel)

    # 3) Determine updates
    updated = 0
    unchanged = 0
    not_found = 0
    ambiguous = 0

    for h, db_entries in db_by_hash.items():
        disk_paths = disk_by_hash.get(h)

        if not disk_paths:
            # hash in DB not found on disk
            not_found += len(db_entries)
            continue

        if len(disk_paths) > 1:
            # same hash exists in multiple places on disk (duplicate copies)
            ambiguous += len(db_entries)
            continue

        new_path = disk_paths[0]

        for reading_id, old_path in db_entries:
            if old_path == new_path:
                unchanged += 1
                continue

            updated += 1
            if args.dry_run:
                print(f"WOULD UPDATE reading_id={reading_id}: {old_path} -> {new_path}")
            else:
                cur.execute(
                    "UPDATE measurement_run SET ide_file_path = ? WHERE reading_id = ?",
                    (new_path, reading_id),
                )

    if not args.dry_run:
        conn.commit()

    print(
        f"\nDone. Updated paths: {updated}. Unchanged: {unchanged}. "
        f"Not found on disk: {not_found}. Ambiguous (duplicate copies): {ambiguous}."
    )


if __name__ == "__main__":
    main()
