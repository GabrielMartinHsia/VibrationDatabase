from pathlib import Path
import sqlite3

from vibtool.db import load_config
from vibtool.endaq_helpers import load_xyz_from_ide


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    data_root = Path(cfg["data_root"])

    # Grab just ONE measurement_run row (first one)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT reading_id, ide_file_path
        FROM measurement_run
        ORDER BY reading_id
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        print("No measurement_run rows found.")
        return

    reading_id, rel_path = row
    ide_path = data_root / rel_path

    print(f"Testing reading_id={reading_id}, file={ide_path}")

    data_x, data_y, data_z, fs = load_xyz_from_ide(ide_path)

    print(f"Sample rate: {fs} Hz")
    print(f"X len: {len(data_x)}, Y len: {len(data_y)}, Z len: {len(data_z)}")


if __name__ == "__main__":
    main()
