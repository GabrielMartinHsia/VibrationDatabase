from pathlib import Path

from vibtool.db import load_config, get_connection
from vibtool.endaq_helpers import load_xyz_from_ide
from vibtool.fft_helpers import compute_fft, array_to_blob


def insert_fft_spectrum(
    conn,
    reading_id: int,
    axis: str,
    sig,
    fs: float,
    window: str = "hann",
    spectrum_type: str = "amplitude",
):
    """
    Compute FFT for one axis and insert into fft_spectrum as a BLOB.
    """
    import numpy as np

    n = len(sig)
    freqs, mag = compute_fft(sig, fs)

    blob = array_to_blob(mag)

    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO fft_spectrum (
            reading_id,
            axis,
            nfft,
            sample_rate_hz,
            window,
            spectrum_type,
            spectrum_blob
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (reading_id, axis, n, fs, window, spectrum_type, blob),
    )
    conn.commit()


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    data_root = Path(cfg["data_root"])

    conn = get_connection(db_path)
    cur = conn.cursor()

    # Select all runs that DON'T yet have any fft_spectrum rows
    cur.execute(
        """
        SELECT mr.reading_id, mr.ide_file_path
        FROM measurement_run mr
        LEFT JOIN fft_spectrum fs ON fs.reading_id = mr.reading_id
        WHERE fs.reading_id IS NULL
        ORDER BY mr.reading_id
        """
    )

    rows = cur.fetchall()
    if not rows:
        print("No measurement_run rows without FFTs. Nothing to do.")
        return

    print(f"Found {len(rows)} runs without FFTs.")

    for reading_id, rel_path in rows:
        ide_path = data_root / rel_path
        print(f"\nProcessing reading_id={reading_id}, file={ide_path}")

        # Load raw data from the IDE
        x, y, z, fs = load_xyz_from_ide(ide_path)
        print(f"  Sample rate: {fs:.2f} Hz, N = {len(x)}")

        # Store FFT for each axis
        insert_fft_spectrum(conn, reading_id, "X", x, fs)
        insert_fft_spectrum(conn, reading_id, "Y", y, fs)
        insert_fft_spectrum(conn, reading_id, "Z", z, fs)

        print("  Stored FFTs for axes X, Y, Z.")


if __name__ == "__main__":
    main()
