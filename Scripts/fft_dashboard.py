from pathlib import Path
import sys
import numpy as np

# Ensure project root (the folder containing 'vibtool') is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sqlite3
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from vibtool.db import load_config, infer_sort_key_from_ide_path
from vibtool.fft_helpers import blob_to_array

import re
from collections import defaultdict


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

def get_all_readings_for_pump(conn, site, pump_name):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT mr.reading_id, mr.location_id, mr.timestamp_utc, mr.ide_file_path
        FROM measurement_run mr
        JOIN measurement_location ml ON mr.location_id = ml.location_id
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ?
        ORDER BY mr.timestamp_utc DESC;
        """,
        (site, pump_name),
    )
    return cur.fetchall()


_NEW_FMT = re.compile(r"^(?P<yymmdd>\d{6})_R(?P<run>\d{1,6})_L\d{2}$", re.IGNORECASE)
_OLD_FMT = re.compile(r"^(?P<yymmdd>\d{6})_(?P<hhmmss>\d{6})_L\d{2}$", re.IGNORECASE)

def compute_run_key(ide_file_path: str) -> str | None:
    p = Path(ide_file_path)
    stem = p.stem

    m = _NEW_FMT.match(stem)
    if m:
        run_num = int(m.group("run"))  # works for R1, R01, R001
        return f"{m.group('yymmdd')}_R{run_num:02d}"

    m = _OLD_FMT.match(stem)
    if m:
        return m.group("yymmdd")

    return None


# def build_runs_index(reading_rows):
#     """
#     Returns:
#       runs: dict[run_key] -> list[sqlite3.Row]
#       run_order: list[run_key] sorted newest->oldest by max timestamp_utc
#     """
#     runs = defaultdict(list)
#     newest_ts = {}

#     for r in reading_rows:
#         rk = compute_run_key(r["ide_file_path"])
#         if rk is None:
#             continue
#         runs[rk].append(r)
#         ts = r["timestamp_utc"]
#         if rk not in newest_ts or ts > newest_ts[rk]:
#             newest_ts[rk] = ts

#     run_order = sorted(newest_ts.keys(), key=lambda k: newest_ts[k], reverse=True)
#     return runs, run_order
def build_runs_index(reading_rows):
    """
    Returns:
      runs: dict[run_key] -> list[sqlite3.Row]
      run_order: list[run_key] sorted newest->oldest by run_key (YYMMDD_R##)
    """
    runs = defaultdict(list)

    for r in reading_rows:
        rk = compute_run_key(r["ide_file_path"])
        if rk is None:
            continue
        runs[rk].append(r)

    # Sort by the run_key itself (works because rk is YYMMDD_R##)
    run_order = sorted(runs.keys(), reverse=True)
    return runs, run_order



def get_latest_reading_for_location(conn, location_id: int):
    """
    Choose 'latest' based on filename convention (YYMMDD_R## preferred),
    because device timestamps may be unreliable.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT reading_id, location_id, timestamp_utc, ide_file_path
        FROM measurement_run
        WHERE location_id = ?
          AND COALESCE(measurement_type, 'route') = 'route'
        """,
        (location_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return None

    # rows: (reading_id, location_id, timestamp_utc, ide_file_path)
    best = None
    best_key = None

    for reading_id, loc_id, ts, ide_path in rows:
        k = infer_sort_key_from_ide_path(ide_path)
        # fallback so we always pick something deterministic
        if k is None:
            k = (-1, -1, 9)

        # include reading_id as a final tie-breaker
        full_key = (k[0], k[1], -k[2], reading_id)

        if best is None or full_key > best_key:
            best = {
                "reading_id": reading_id,
                "location_id": loc_id,
                "timestamp_utc": ts,
                "ide_file_path": ide_path,
            }
            best_key = full_key

    return best


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


def accel_mag_g_to_velocity_mag_in_s(mag_g: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """
    Convert acceleration spectrum magnitude in g (0-pk, amplitude spectrum)
    to velocity spectrum magnitude in in/s (0-pk) via V = A / (2*pi*f).
    """
    g_to_m_s2 = 9.81
    m_to_mm = 1000.0
    mm_to_in = 1.0 / 25.4

    f = freqs_hz.astype(float).copy()
    f[0] = np.inf  # avoid divide-by-zero at DC

    vel = (mag_g * g_to_m_s2) / (2.0 * np.pi * f) * m_to_mm * mm_to_in
    vel[0] = 0.0
    return vel


def transform_fft_for_display(freqs: np.ndarray, mag_g: np.ndarray, display_mode: str) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Returns (freqs, mag, y_label)
    mag input is acceleration magnitude in g (0-pk) from DB.
    """
    if display_mode.startswith("Velocity"):
        g_to_m_s2 = 9.81
        m_to_mm = 1000.0
        mm_to_in = 1.0 / 25.4

        f = freqs.astype(float).copy()
        f[0] = np.inf
        mag = (mag_g * g_to_m_s2) / (2.0 * np.pi * f) * m_to_mm * mm_to_in
        mag[0] = 0.0
        return freqs, mag, "Velocity (in/s) (0-pk)"

    return freqs, mag_g, "Acceleration (g) (0-pk)"


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

    loc_label_map = {
        row["location_id"]: (
            f"{row['name']} — {row['description']}".strip()
            if row["description"]
            else row["name"]
        )
        for row in loc_rows
    }

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
        ["Latest per location", "Select run", "Compare two runs"],
        index=0,
    )

    display_mode = st.sidebar.radio(
        "Display",
        ["Acceleration (g)", "Velocity (in/s)"],
        index=0,
    )

    y_label = "Velocity (in/s) (0-pk)" if display_mode.startswith("Velocity") else "Acceleration (g) (0-pk)"

    all_readings = get_all_readings_for_pump(conn, site, pump_name)
    runs, run_order = build_runs_index(all_readings)

    selected_run = None
    run_a = None
    run_b = None

    if mode in ("Select run", "Compare two runs"):
        if not run_order:
            st.sidebar.error("No run keys could be derived from filenames. (Check naming convention YYMMDD_RXX_LXX.)")
            return

        # Nice labels showing newest timestamp
        # (We already have readings sorted by timestamp desc, but we’ll compute labels simply.)
        run_labels = run_order

        if mode == "Select run":
            selected_run = st.sidebar.selectbox("Run", run_labels, index=0)

        if mode == "Compare two runs":
            run_a = st.sidebar.selectbox("Run A", run_labels, index=0)
            run_b = st.sidebar.selectbox("Run B", run_labels, index=min(1, len(run_labels)-1))


    st.sidebar.markdown("---")
    st.sidebar.caption("FFT data from vibes.db")

    # Main plotting area
    if not selected_loc_ids:
        st.info("Select at least one location to plot.")
        return

    fig = go.Figure()
    n_traces = 0

    used_run_keys = set()

    for loc_id in selected_loc_ids:
        loc_name = loc_name_map[loc_id]
        orient_info = loc_orient_map[loc_id]
        trace_label = loc_label_map.get(loc_id, loc_name)

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
                f" {loc_name}: no device axis mapped to '{desired_orientation}', skipping."
            )
            continue

        axis = orientation_to_axis[desired_orientation]


        def pick_reading_for_location_from_run(run_key: str, location_id: int):
            # If there are multiple (shouldn't be), pick the newest timestamp_utc
            candidates = [r for r in runs.get(run_key, []) if r["location_id"] == location_id]
            if not candidates:
                return None
            return max(candidates, key=lambda r: r["timestamp_utc"])


        if mode == "Latest per location":
            reading = get_latest_reading_for_location(conn, loc_id)
            if reading is None:
                st.write(f"{loc_name}: no readings found, skipping.")
                continue

            rk = compute_run_key(reading["ide_file_path"])
            if rk:
                used_run_keys.add(rk)

            reading_id = reading["reading_id"]

            fft_data = get_fft_for_reading(conn, reading_id, axis)
            if fft_data is None:
                st.write(f" {loc_name}: no FFT for reading {reading_id}, axis {axis}, skipping.")
                continue

            freqs, mag, fs, nfft = fft_data
            freqs, mag, y_label = transform_fft_for_display(freqs, mag, display_mode)
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            if not mask.any():
                st.write(f"{loc_name}: no frequencies between {min_freq} and {max_freq} Hz, skipping.")
                continue

            fig.add_trace(go.Scatter(
                x=freqs[mask],
                y=mag[mask],
                mode="lines",
                # name=f"{loc_name} ({desired_orientation}) — {Path(reading['ide_file_path']).stem}",
                name=trace_label,
            ))
            n_traces += 1

        elif mode == "Select run":
            r = pick_reading_for_location_from_run(selected_run, loc_id)
            if r is None:
                st.write(f"{loc_name}: no reading found in run {selected_run}, skipping.")
                continue

            reading_id = r["reading_id"]
            timestamp = r["timestamp_utc"]

            fft_data = get_fft_for_reading(conn, reading_id, axis)
            if fft_data is None:
                st.write(f" {loc_name}: no FFT for reading {reading_id}, axis {axis}, skipping.")
                continue

            freqs, mag, fs, nfft = fft_data
            freqs, mag, y_label = transform_fft_for_display(freqs, mag, display_mode)
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            if not mask.any():
                st.write(f"{loc_name}: no frequencies between {min_freq} and {max_freq} Hz, skipping.")
                continue

            fig.add_trace(go.Scatter(
                x=freqs[mask],
                y=mag[mask],
                mode="lines",
                # name=f"{loc_name} ({desired_orientation}) — {selected_run}",
                name=trace_label,
            ))
            n_traces += 1

        elif mode == "Compare two runs":
            for rk in (run_a, run_b):
                r = pick_reading_for_location_from_run(rk, loc_id)
                if r is None:
                    # Don't spam warnings twice; just skip missing combos
                    continue

                reading_id = r["reading_id"]
                fft_data = get_fft_for_reading(conn, reading_id, axis)
                if fft_data is None:
                    continue

                freqs, mag, fs, nfft = fft_data
                freqs, mag, y_label = transform_fft_for_display(freqs, mag, display_mode)
                mask = (freqs >= min_freq) & (freqs <= max_freq)
                if not mask.any():
                    continue

                fig.add_trace(go.Scatter(
                    x=freqs[mask],
                    y=mag[mask],
                    mode="lines",
                    # name=f"{loc_name} ({desired_orientation}) — {rk}",
                    name=f"{trace_label} — {rk}",
                ))
                n_traces += 1


    if n_traces == 0:
        st.warning("No FFT traces to display with the current selection.")
        return

    if mode == "Select run":
        run_info = f"Run: {selected_run}"
    elif mode == "Compare two runs":
        run_info = f"Runs: {run_a} vs {run_b}"
    else:  # Latest per location
        if len(used_run_keys) == 1:
            run_info = f"Latest run: {next(iter(used_run_keys))}"
        elif len(used_run_keys) > 1:
            run_info = f"Latest runs: {', '.join(sorted(used_run_keys))}"
        else:
            run_info = "Latest per location"

    title = f"FFT Comparison — {site} / {pump_name} — {desired_orientation.capitalize()} — {run_info}"

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title=y_label,
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
