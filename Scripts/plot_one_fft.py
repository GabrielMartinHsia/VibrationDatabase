from pathlib import Path
import sqlite3
import numpy as np
import plotly.graph_objects as go

from vibtool.db import load_config
from vibtool.fft_helpers import blob_to_array


def get_fft(conn, reading_id, axis):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT nfft, sample_rate_hz, spectrum_blob
        FROM fft_spectrum
        WHERE reading_id = ? AND axis = ?
        """,
        (reading_id, axis),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"No FFT found for reading_id={reading_id}, axis={axis}")

    nfft, fs, blob = row
    mag = blob_to_array(blob)

    # reconstruct frequency axis
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)

    return freqs, mag


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    conn = sqlite3.connect(db_path)

    # Pick what you want to plot:
    reading_id = 1
    axis = "X"

    freqs, mag = get_fft(conn, reading_id, axis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=mag, mode="lines", name=f"Run {reading_id} - Axis {axis}"))
    fig.update_layout(
        title=f"FFT Spectrum for Run {reading_id}, Axis {axis}",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (g)",
        template="plotly_white",
    )
    fig.show()


if __name__ == "__main__":
    main()
