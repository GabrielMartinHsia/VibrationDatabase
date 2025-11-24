from pathlib import Path
import sqlite3

from vibtool.db import load_config
from vibtool.endaq_helpers import load_xyz_from_ide
from vibtool.fft_helpers import compute_fft


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    data_root = Path(cfg["data_root"])

    # Grab the first measurement_run
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

    print(f"Testing FFT for reading_id={reading_id}, file={ide_path}")

    x, y, z, fs = load_xyz_from_ide(ide_path)
    print(f"Sample rate: {fs} Hz, N = {len(x)} samples")

    freqs, mag_x = compute_fft(x, fs)

    print(f"FFT length: {len(freqs)} frequency bins")
    print("First 10 freq bins (Hz, mag):")
    for f, m in list(zip(freqs, mag_x))[:10]:
        print(f"  {f:.1f} Hz  ->  {m:.6g}")


if __name__ == "__main__":
    main()
