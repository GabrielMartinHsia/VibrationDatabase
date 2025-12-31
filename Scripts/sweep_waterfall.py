from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.colors as pc

from vibtool.db import load_config, get_connection
from vibtool.endaq_helpers import load_xyz_from_ide


try:
    from scipy.signal import spectrogram, get_window
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------------- DB helpers ----------------

def get_sites(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT site FROM pump ORDER BY site")
    return [r[0] for r in cur.fetchall()]

def get_machines(conn, site: str):
    cur = conn.cursor()
    cur.execute("SELECT name FROM pump WHERE site = ? ORDER BY name", (site,))
    return [r[0] for r in cur.fetchall()]

def get_locations_for_machine(conn, site: str, machine: str):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT ml.location_id, ml.name
        FROM measurement_location ml
        JOIN pump p ON ml.pump_id = p.pump_id
        WHERE p.site = ? AND p.name = ?
        ORDER BY ml.name
        """,
        (site, machine),
    )
    return cur.fetchall()

def get_sweeps_for_location(conn, location_id: int):
    cur = conn.cursor()
    cur.execute(
        """
        SELECT reading_id, timestamp_utc, ide_file_path
        FROM measurement_run
        WHERE location_id = ?
          AND COALESCE(measurement_type, 'route') = 'sweep'
        ORDER BY timestamp_utc DESC
        """,
        (location_id,),
    )
    return cur.fetchall()


# ---------------- IDE loading ----------------

@st.cache_data(show_spinner=False)
def load_ide_timeseries(abs_path: str):
    """
    Return (t_seconds, axes_dict, fs_hz)
      axes_dict: {"X": np.ndarray, "Y": np.ndarray, "Z": np.ndarray} in g
    """
    ide_path = Path(abs_path)
    x, y, z, fs = load_xyz_from_ide(ide_path)

    n = len(x)
    t = np.arange(n, dtype=float) / float(fs)

    axes = {"X": np.asarray(x), "Y": np.asarray(y), "Z": np.asarray(z)}
    return t, axes, float(fs)


# ---------------- Cached STFT Function -----------------
@st.cache_data(show_spinner=False)
def cached_stft(abs_path: str, axis: str, win_s: float, overlap_pct: int):
    """
    Compute STFT magnitude once per (file, axis, win_s, overlap_pct).
    Returns freqs_full, times_full, mag_full, fs
    """
    t, axes, fs = load_ide_timeseries(abs_path)
    x = axes[axis]
    freqs, times, mag = stft_magnitude(x, fs=float(fs), win_s=float(win_s), overlap_pct=int(overlap_pct))
    return freqs, times, mag, float(fs)


# ---------------- Waterfall computation ----------------

def stft_magnitude(x: np.ndarray, fs: float, win_s: float, overlap_pct: int):
    """
    Returns:
      freqs (Hz), times (s), mags (freq x time)  [linear magnitude]
    """
    x = x.astype(float)
    x = x - np.mean(x)

    nperseg = max(16, int(win_s * fs))
    noverlap = int(nperseg * overlap_pct / 100)
    hop = max(1, nperseg - noverlap)

    if HAVE_SCIPY:
        # Use scipy.signal.spectrogram in magnitude mode (STFT mag)
        f, t, S = spectrogram(
            x,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            detrend=False,
            scaling="density",
            mode="magnitude",
        )
        return f, t, S

    # Numpy fallback
    window = np.hanning(nperseg)
    frames = 1 + max(0, (len(x) - nperseg) // hop)

    mags = []
    times = []
    for i in range(frames):
        start = i * hop
        seg = x[start:start+nperseg]
        if len(seg) < nperseg:
            break
        seg = (seg - np.mean(seg)) * window
        X = np.fft.rfft(seg)
        mag = np.abs(X) / nperseg
        mag *= 2.0  # one-sided scale
        mags.append(mag)
        times.append((start + nperseg / 2) / fs)

    mags = np.array(mags).T  # freq x time
    freqs = np.fft.rfftfreq(nperseg, d=1/fs)
    times = np.array(times)
    return freqs, times, mags


def accel_to_velocity_in_s(mag_accel_g: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """
    Convert acceleration spectrum magnitude [g] to velocity spectrum magnitude [mm/s]
    using V = A / (2*pi*f), with A in m/s^2.
    mag_accel_g: (freq x time)
    freqs_hz: (freq,)
    """
    g_to_m_s2 = 9.81
    m_to_mm = 1000.0
    mm_to_in = 1 / 25.4

    safe_f = freqs_hz.copy().astype(float)
    safe_f[0] = np.inf

    vel = (mag_accel_g * g_to_m_s2) / (2 * np.pi * safe_f[:, None]) * m_to_mm * mm_to_in
    vel[0, :] = 0.0
    return vel


# def downsample_2d(freqs, times, Z, max_time_slices: int, max_freq_bins: int):
#     """
#     Downsample the waterfall for responsiveness.
#     """
#     # time downsample
#     if Z.shape[1] > max_time_slices:
#         step = int(np.ceil(Z.shape[1] / max_time_slices))
#         times = times[::step]
#         Z = Z[:, ::step]

#     # freq downsample
#     if Z.shape[0] > max_freq_bins:
#         step = int(np.ceil(Z.shape[0] / max_freq_bins))
#         freqs = freqs[::step]
#         Z = Z[::step, :]

#     return freqs, times, Z

# def downsample_freq_stable(freqs_full, freqs_view, Z_view, max_freq_bins: int):
#     """
#     Downsample frequency using a stride based on the FULL spectrum length,
#     so the stride doesn't change dramatically as you adjust fmin/fmax.
#     """
#     if max_freq_bins <= 0:
#         return freqs_view, Z_view

#     stride = int(np.ceil(len(freqs_full) / max_freq_bins))
#     stride = max(1, stride)

#     return freqs_view[::stride], Z_view[::stride, :]


def downsample_freq_maxhold(freqs_full, freqs_view, Z_view, max_freq_bins: int):
    """
    Downsample frequency by grouping neighboring bins and taking MAX within each group.
    Preserves narrow peaks better than simple slicing.
    """
    if max_freq_bins <= 0 or len(freqs_view) <= max_freq_bins:
        return freqs_view, Z_view

    # stable group size based on full-spectrum length
    group = int(np.ceil(len(freqs_full) / max_freq_bins))
    group = max(1, group)

    n = len(freqs_view)
    out_freqs = []
    out_Z = []

    for i in range(0, n, group):
        j = min(i + group, n)
        out_freqs.append(freqs_view[i:j].mean())
        out_Z.append(np.max(Z_view[i:j, :], axis=0))

    return np.array(out_freqs), np.vstack(out_Z)


def downsample_time_only(freqs, times, Z, max_time_slices: int):
    if Z.shape[1] > max_time_slices:
        step = int(np.ceil(Z.shape[1] / max_time_slices))
        times = times[::step]
        Z = Z[:, ::step]
    return freqs, times, Z



def make_heatmap_figure(freqs, times, Z, title: str, z_label: str):
    fig = go.Figure(
        data=go.Heatmap(
            x=times,
            y=freqs,
            z=Z,
            colorbar=dict(title=z_label),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Frequency (Hz)",
        height=800,
        margin=dict(t=70, b=50, l=70, r=70),
    )
    return fig


def make_surface_figure(freqs, times, Z, white_floor_pct, title: str, z_label: str):
    # Meshgrid expects X, Y shaped like Z
    X, Y = np.meshgrid(times, freqs)

    colorscale = make_white_floor_colorscale(
        Z,
        base_cmap="viridis",
        white_floor_pct=white_floor_pct,
    )

    fig = go.Figure(
        data=[go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=colorscale,
            colorbar=dict(title=z_label),
        )]
    )


    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            zaxis_title="",
            zaxis=dict(showticklabels=False),
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False),
            zaxis_backgroundcolor="white",
            bgcolor="white",
            aspectmode="manual",
            aspectratio=dict(x=1.8, y=1.0, z=0.7),
        ),
        scene_camera=dict(eye=dict(x=0.0, y=-1.75, z=0.0)),
        height=800,
    )
    return fig


def make_white_floor_colorscale(Z, base_cmap="viridis", white_floor_pct=2.0):
    """
    Create a Plotly colorscale where the lowest values are white.
    white_floor_pct: percentage of max(Z) below which color is white
    """
    Zmax = np.nanmax(Z)
    if Zmax <= 0:
        return pc.get_colorscale(base_cmap)

    # Normalize cutoff into [0,1] color scale
    z_cut = (white_floor_pct / 100.0)
    z_cut = min(max(z_cut, 0.0), 0.99)

    base = pc.get_colorscale(base_cmap)

    colorscale = [
        [0.0, "rgb(255,255,255)"],
        [z_cut, "rgb(255,255,255)"],
    ]

    for frac, color in base:
        adj_frac = z_cut + (1.0 - z_cut) * frac
        colorscale.append([adj_frac, color])

    return colorscale



# ---------------- Streamlit UI ----------------

def main():
    st.set_page_config(page_title="Sweep Waterfall (3D)", layout="wide")
    st.title("Sweep Waterfall Viewer (3D)")

    cfg = load_config()
    data_root = Path(cfg["data_root"])
    conn = get_connection(cfg["db_path"])

    sites = get_sites(conn)
    if not sites:
        st.error("No sites found in DB.")
        return

    site = st.sidebar.selectbox("Site", sites, index=0)
    machines = get_machines(conn, site)
    machine = st.sidebar.selectbox("Machine", machines, index=0)

    loc_rows = get_locations_for_machine(conn, site, machine)
    loc_label = {loc_id: name for loc_id, name in loc_rows}
    # loc_id = st.sidebar.selectbox("Location", [r[0] for r in loc_rows], format_func=lambda i: loc_label[i])
    loc_ids = [r[0] for r in loc_rows]
    loc_id = st.sidebar.selectbox(
        "Location",
        loc_ids,
        format_func=lambda i: loc_label.get(i, f"location_id={i}"),
        key=f"loc_select_{site}_{machine}",  # <-- important: new state per machine
    )

    if loc_id not in loc_label:
        st.warning("Selected location is not valid for this machine. Resetting selection.")
        loc_id = loc_ids[0]


    sweeps = get_sweeps_for_location(conn, loc_id)
    if not sweeps:
        st.info("No sweep recordings found for this location (measurement_type='sweep').")
        return

    def fmt_sweep(r):
        reading_id, ts, path = r
        return f"{ts} — {Path(path).name} (id={reading_id})"

    # reading_id, ts, rel_path = st.sidebar.selectbox("Sweep recording", sweeps, format_func=fmt_sweep)
    reading_id, ts, rel_path = st.sidebar.selectbox(
        "Sweep recording",
        sweeps,
        format_func=fmt_sweep,
        key=f"sweep_select_{site}_{machine}_{loc_id}",
    )


    axis = st.sidebar.selectbox("Axis", ["X", "Y", "Z"], index=2)
    data_type = st.sidebar.radio("Display", ["Acceleration (g)", "Velocity (in/s)"], index=0)

    view_mode = st.sidebar.radio("View", ["2D Heatmap", "3D Surface"], index=1)

    white_floor_pct = st.sidebar.slider(
        "White floor (% of max amplitude)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Values below this percentage of max are rendered white",
    )


    dc_floor = st.sidebar.selectbox("DC floor preset", [0.0, 0.5, 1.0, 2.0, 5.0], index=3)

    fmin = st.sidebar.number_input(
        "Min frequency (Hz)",
        min_value=0.0,
        max_value=500.0,   # keep this small because it's for low-end precision
        value=float(dc_floor),
        step=0.1,
        format="%.1f",
    )

    fmax = st.sidebar.slider(
        "Max frequency (Hz)",
        min_value=10.0,
        max_value=5000.0,
        value=2000.0,
        step=10.0,
    )


    win_s = st.sidebar.select_slider("Window (s)", options=[0.25, 0.5, 1.0, 2.0, 4.0], value=1.0)
    overlap_pct = st.sidebar.slider("Overlap (%)", 0, 95, 75, 5)

    # Responsiveness controls (important for 3D)
    max_time_slices = st.sidebar.slider("Max time slices (performance)", 50, 400, 180, 10)
    max_freq_bins = st.sidebar.slider("Max freq bins (performance)", 100, 1200, 1200, 50)

    abs_path = data_root / rel_path
    if not abs_path.exists():
        st.error(f"IDE file not found on disk: {abs_path}")
        return

    # st.caption(f"**{site}/{machine}** — {loc_label[loc_id]} — {Path(rel_path).name}")
    st.caption(f"**{site}/{machine}** — {loc_label.get(loc_id, loc_id)} — {Path(rel_path).name}")


    with st.spinner("Loading IDE time series..."):
        t, axes, fs = load_ide_timeseries(str(abs_path))

    if axis not in axes:
        st.error(f"Axis {axis} not available in IDE.")
        return

    x = axes[axis]


    
    with st.spinner("Computing / loading STFT (waterfall)..."):
        # Compute once (cached) per file+axis+window+overlap
        freqs_full, times_full, mag_full, fs = cached_stft(
            str(abs_path),
            axis,
            float(win_s),
            int(overlap_pct),
        )

        # Clip view band (cheap)
        mask = (freqs_full >= float(fmin)) & (freqs_full <= float(fmax))
        freqs_view = freqs_full[mask]
        mag_view = mag_full[mask, :]

        # Convert (cheap)
        if data_type.startswith("Velocity"):
            Z_view = accel_to_velocity_in_s(mag_view, freqs_view)
            z_label = "in/s (0-pk)"
        else:
            Z_view = mag_view
            z_label = "g (0-pk)"

        # Downsample TIME (cheap)
        freqs_view, times_view, Z_view = downsample_time_only(
            freqs_view, times_full, Z_view, max_time_slices=max_time_slices
        )

        # Downsample FREQ with stable stride (cheap + consistent)
        freqs_view, Z_view = downsample_freq_maxhold(
            freqs_full=freqs_full,
            freqs_view=freqs_view,
            Z_view=Z_view,
            max_freq_bins=max_freq_bins,
        )

        # Final outputs used by plots
        freqs, times, Z = freqs_view, times_view, Z_view


    title = f"Waterfall — {site}/{machine} {loc_label[loc_id]} — {axis} — {data_type} — {fmin:.1f}-{fmax:.1f} Hz"

    if view_mode == "3D Surface":
        fig = make_surface_figure(freqs, times, Z, white_floor_pct, title=title, z_label=z_label)
    else:
        fig = make_heatmap_figure(freqs, times, Z, title=title, z_label=z_label)

    st.plotly_chart(fig, use_container_width=True)


    with st.expander("Details"):
        st.write({
            "reading_id": int(reading_id),
            "timestamp_utc": ts,
            "fs_hz": float(fs),
            "window_s": float(win_s),
            "overlap_pct": int(overlap_pct),
            "freq_bins": int(len(freqs)),
            "time_slices": int(len(times)),
            "scipy_available": HAVE_SCIPY,
        })


if __name__ == "__main__":
    main()
