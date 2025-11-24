import io
import numpy as np


def compute_fft(signal, fs):
    """
    Compute single-sided magnitude FFT for a real-valued time series.

    Args:
        signal: 1D NumPy array
        fs: sample rate [Hz]

    Returns:
        freqs: frequency axis [Hz] for the positive frequencies
        mag: magnitude spectrum (same length as freqs)
    """
    signal = np.asarray(signal)

    # Remove DC offset (mean) to kill the big spike at ~0 Hz
    signal = signal - np.mean(signal)

    n = len(signal)
    # Hann window to reduce spectral leakage
    window = np.hanning(n)
    sig_win = signal * window

    # Real FFT
    fft_vals = np.fft.rfft(sig_win)
    mag = np.abs(fft_vals) * 2.0 / np.sum(window)

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, mag


def array_to_blob(arr: np.ndarray) -> bytes:
    """
    Serialize a NumPy array to bytes suitable for storing in a SQLite BLOB.
    Uses np.save into an in-memory buffer.
    """
    buf = io.BytesIO()
    # Use float32 to save space; adjust if you want more precision
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def blob_to_array(blob: bytes) -> np.ndarray:
    """
    Deserialize a SQLite BLOB (created by array_to_blob) back into a NumPy array.
    """
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf)
