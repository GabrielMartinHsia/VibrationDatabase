from pathlib import Path
import sqlite3
import numpy as np

from vibtool.db import load_config
from vibtool.fft_helpers import blob_to_array


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Grab one FFT row
    cur.execute(
        """
        SELECT reading_id, axis, nfft, sample_rate_hz, spectrum_blob
        FROM fft_spectrum
        ORDER BY reading_id, axis
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if not row:
        print("No FFT rows found.")
        return

    reading_id, axis, nfft, fs, blob = row
    mag = blob_to_array(blob)

    print(f"reading_id={reading_id}, axis={axis}, nfft={nfft}, fs={fs}")
    print(f"mag length = {len(mag)}")

    # Reconstruct frequency axis for this FFT
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    print(f"freqs length = {len(freqs)}")
    print("First 10 (freq, mag):")
    for f, m in list(zip(freqs, mag))[:10]:
        print(f"  {f:.1f} Hz  ->  {m:.6g}")


if __name__ == "__main__":
    main()
