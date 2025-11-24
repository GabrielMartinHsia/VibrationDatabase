from pathlib import Path
import sqlite3
import numpy as np
import plotly.graph_objects as go

from vibtool.db import load_config
from vibtool.fft_helpers import blob_to_array


def get_latest_readings_for_pump(conn, site: str, pump_name: str):
    """
    For a given site/pump, return the latest reading per location.

    Returns a list of dicts:
      {
        "reading_id": ...,
        "timestamp_utc": ...,
        "loc_name": ...,
        "loc_desc": ...,
        "x_dir": ...,
        "y_dir": ...,
        "z_dir": ...,
        "pump_name": ...,
        "site_name": ...
      }
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            mr.reading_id,
            mr.timestamp_utc,
            ml.location_id,
            ml.name AS loc_name,
            ml.description AS loc_desc,
            ml.x_dir,
            ml.y_dir,
            ml.z_dir,
            p.name AS pump_name,
            p.site AS site_name
        FROM measurement_run mr
        JOIN measurement_location ml ON mr.location_id = ml.location_id
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ?
        ORDER BY ml.location_id, mr.timestamp_utc DESC
        """,
        (site, pump_name),
    )

    rows = cur.fetchall()
    latest_by_loc = {}
    for (
        reading_id,
        timestamp_utc,
        location_id,
        loc_name,
        loc_desc,
        x_dir,
        y_dir,
        z_dir,
        pump_name,
        site_name,
    ) in rows:
        if location_id in latest_by_loc:
            continue  # we already took the latest (first in DESC order)
        latest_by_loc[location_id] = {
            "reading_id": reading_id,
            "timestamp_utc": timestamp_utc,
            "loc_name": loc_name,
            "loc_desc": loc_desc,
            "x_dir": x_dir,
            "y_dir": y_dir,
            "z_dir": z_dir,
            "pump_name": pump_name,
            "site_name": site_name,
        }

    return list(latest_by_loc.values())


def get_fft_for_reading(conn, reading_id: int, axis: str):
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
        return None

    nfft, fs, blob = row
    mag = blob_to_array(blob)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag, fs, nfft


def main():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    conn = sqlite3.connect(db_path)

    # ---- choose what to compare ----
    site = "Red"
    pump_name = "P34A"
    desired_orientation = "horizontal"   # 'horizontal', 'vertical', 'axial'
    max_freq = 5000.0                    # Hz, or None for full spectrum
    # -------------------------------

    readings = get_latest_readings_for_pump(conn, site, pump_name)
    print(f"Found {len(readings)} candidate locations for {site} / {pump_name}")

    if not readings:
        print(f"No readings found for {site} / {pump_name}")
        return

    fig = go.Figure()
    n_traces = 0

    for r in readings:
        reading_id = r["reading_id"]
        x_dir = (r["x_dir"] or "").lower()
        y_dir = (r["y_dir"] or "").lower()
        z_dir = (r["z_dir"] or "").lower()

        # Build orientation → device axis map
        orientation_to_axis = {}
        if x_dir:
            orientation_to_axis[x_dir] = "X"
        if y_dir:
            orientation_to_axis[y_dir] = "Y"
        if z_dir:
            orientation_to_axis[z_dir] = "Z"

        if desired_orientation.lower() not in orientation_to_axis:
            print(
                f"\nReading {reading_id} at {r['loc_name']}: "
                f"no device axis mapped to orientation '{desired_orientation}', skipping."
            )
            continue

        axis = orientation_to_axis[desired_orientation.lower()]
        print(f"\nReading {reading_id} at {r['loc_name']}: using device axis {axis} for {desired_orientation}")

        fft_data = get_fft_for_reading(conn, reading_id, axis)
        if fft_data is None:
            print(f"  No FFT for reading {reading_id}, axis {axis}, skipping.")
            continue

        freqs, mag, fs, nfft = fft_data
        print(f"  fs={fs:.3f} Hz, nfft={nfft}, fft_len={len(freqs)}")

        freqs_plot = freqs
        mag_plot = mag
        if max_freq is not None:
            mask = freqs <= max_freq
            if not mask.any():
                print(f"  No frequencies <= {max_freq} Hz, skipping.")
                continue
            freqs_plot = freqs[mask]
            mag_plot = mag[mask]
            print(f"  After max_freq filter: {len(freqs_plot)} points")

        label = f"{r['loc_name']} ({desired_orientation})"

        fig.add_trace(
            go.Scatter(
                x=freqs_plot,
                y=mag_plot,
                mode="lines",
                name=label,
            )
        )
        print(f"  Added trace: {label}")
        n_traces += 1

    if n_traces == 0:
        print("No traces were added to the figure. Nothing to plot.")
        return

    title = f"FFT Comparison — {site} / {pump_name} — {desired_orientation.capitalize()} vibration"

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (g)",
        template="plotly_white",
    )

    fig.show()




if __name__ == "__main__":
    main()
