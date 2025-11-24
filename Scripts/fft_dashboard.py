from pathlib import Path
import sys

# Ensure project root (the folder containing 'vibtool') is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from vibtool.db import load_config
from vibtool.fft_helpers import blob_to_array


# ---------- DB helpers ----------

def get_connection():
    cfg = load_config()
    db_path = Path(cfg["db_path"])
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def get_sites(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT site FROM pump ORDER BY site;")
    return [row["site"] for row in cur.fetchall()]


def get_pumps_for_site(conn, site):
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM pump WHERE site = ? ORDER BY name;",
        (site,),
    )
    return [row["name"] for row in cur.fetchall()]


def get_locations_for_pump(conn, site, pump_name):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ml.location_id, ml.name, ml.description,
               ml.x_dir, ml.y_dir, ml.z_dir
        FROM measurement_location ml
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ?
        ORDER BY ml.location_id;
        """,
        (site, pump_name),
    )
    return cur.fetchall()


def get_latest_reading_for_location(conn, location_id):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT reading_id, timestamp_utc
        FROM measurement_run
        WHERE location_id = ?
        ORDER BY timestamp_utc DESC
        LIMIT 1;
        """,
        (location_id,),
    )
    return cur.fetchone()


def get_fft_for_reading(conn, reading_id, axis):
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
    if row is None:
        return None

    nfft = row["nfft"]
    fs = row["sample_rate_hz"]
    blob = row["spectrum_blob"]
    mag = blob_to_array(blob)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return freqs, mag, fs, nfft


# ---------- App logic ----------

def main():
    st.set_page_config(page_title="Vibration FFT Dashboard", layout="wide")
    st.title("Vibration FFT Dashboard")

    conn = get_connection()

    # Sidebar controls
    st.sidebar.header("Selection")

    sites = get_sites(conn)
    if not sites:
        st.sidebar.error("No sites found in DB.")
        return

    site = st.sidebar.selectbox("Site", sites, index=0)

    pumps = get_pumps_for_site(conn, site)
    if not pumps:
        st.sidebar.error(f"No pumps found for site {site}.")
        return

    pump_name = st.sidebar.selectbox("Pump", pumps, index=0)

    loc_rows = get_locations_for_pump(conn, site, pump_name)
    if not loc_rows:
        st.sidebar.error(f"No locations found for {site} / {pump_name}.")
        return

    loc_labels = [f"{row['name']} — {row['description']}" if row["description"] else row["name"]
                  for row in loc_rows]
    loc_ids = [row["location_id"] for row in loc_rows]
    loc_name_map = {row["location_id"]: row["name"] for row in loc_rows}
    loc_orient_map = {
        row["location_id"]: {
            "x_dir": (row["x_dir"] or "").lower(),
            "y_dir": (row["y_dir"] or "").lower(),
            "z_dir": (row["z_dir"] or "").lower(),
        }
        for row in loc_rows
    }

    selected_labels = st.sidebar.multiselect(
        "Locations",
        options=loc_labels,
        default=loc_labels,  # select all by default
    )

    selected_loc_ids = [
        loc_ids[loc_labels.index(lbl)] for lbl in selected_labels
    ] if selected_labels else []

    orientation_options = ["horizontal", "vertical", "axial"]
    desired_orientation = st.sidebar.selectbox(
        "Orientation (physical)",
        orientation_options,
        index=0,
    )

    min_freq = st.sidebar.number_input(
        "Min frequency to display (Hz)",
        min_value=0.0,
        max_value=1000.0,
        value=5.0,   # default: ignore 0–5 Hz, where DC & drift live
        step=0.5,
    )

    max_freq = st.sidebar.number_input(
        "Max frequency to display (Hz)",
        min_value=10.0,
        max_value=20000.0,
        value=5000.0,
        step=10.0,
    )


    mode = st.sidebar.radio(
        "Reading selection",
        ["Latest per location"],  # we can add more modes later
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("FFT data from vibes.db")

    # Main plotting area
    if not selected_loc_ids:
        st.info("Select at least one location to plot.")
        return

    fig = go.Figure()
    n_traces = 0

    for loc_id in selected_loc_ids:
        loc_name = loc_name_map[loc_id]
        orient_info = loc_orient_map[loc_id]

        # Map desired physical orientation → device axis
        x_dir = orient_info["x_dir"]
        y_dir = orient_info["y_dir"]
        z_dir = orient_info["z_dir"]

        orientation_to_axis = {}
        if x_dir:
            orientation_to_axis[x_dir] = "X"
        if y_dir:
            orientation_to_axis[y_dir] = "Y"
        if z_dir:
            orientation_to_axis[z_dir] = "Z"

        if desired_orientation not in orientation_to_axis:
            st.write(
                f"⚠️ {loc_name}: no device axis mapped to '{desired_orientation}', skipping."
            )
            continue

        axis = orientation_to_axis[desired_orientation]

        if mode == "Latest per location":
            reading = get_latest_reading_for_location(conn, loc_id)
            if reading is None:
                st.write(f"{loc_name}: no readings found, skipping.")
                continue

            reading_id = reading["reading_id"]
            timestamp = reading["timestamp_utc"]
            fft_data = get_fft_for_reading(conn, reading_id, axis)
            if fft_data is None:
                st.write(
                    f"⚠️ {loc_name}: no FFT for reading {reading_id}, axis {axis}, skipping."
                )
                continue

            freqs, mag, fs, nfft = fft_data

            # Apply BOTH min and max frequency limits
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            if not mask.any():
                st.write(
                    f"{loc_name}: no frequencies between {min_freq} and {max_freq} Hz, skipping."
                )
                continue

            freqs_plot = freqs[mask]
            mag_plot = mag[mask]


            freqs_plot = freqs[mask]
            mag_plot = mag[mask]

            label = f"{loc_name} ({desired_orientation}) — {timestamp}"

            fig.add_trace(
                go.Scatter(
                    x=freqs_plot,
                    y=mag_plot,
                    mode="lines",
                    name=label,
                )
            )
            n_traces += 1

    if n_traces == 0:
        st.warning("No FFT traces to display with the current selection.")
        return

    # title = f"FFT Comparison — {site} / {pump_name} — {desired_orientation.capitalize()}"
    # fig.update_layout(
    #     title=title,
    #     xaxis_title="Frequency (Hz)",
    #     yaxis_title="Magnitude (g)",
    #     template="plotly_white",
    #     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    # )

    title = f"FFT Comparison — {site} / {pump_name} — {desired_orientation.capitalize()}"
    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="Magnitude (g)",
        template="plotly_white",
        height=700,  # make the chart taller
        margin=dict(t=80, b=40, l=60, r=200),  # extra top + right for title + legend
        legend=dict(
            orientation="v",    # vertical legend
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=1.02,             # put legend just to the right of the plot
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
