from pathlib import Path
import numpy as np

from endaq.ide import get_doc, get_channel_table, to_pandas


def load_xyz_from_ide(path: str | Path):
    """
    Load three acceleration channels (X, Y, Z) and sample rate from an enDAQ .IDE file.

    Strategy (based on enDAQ examples):
      - Read the channel table.
      - Filter rows where units == 'g' (acceleration).
      - Further filter to those with the maximum number of samples.
      - Take the parent Channel of that SubChannel.
      - Use endaq.ide.to_pandas() on the parent Channel to get a time-series DataFrame.
      - Select the columns whose names match the SubChannel 'name' values
        (e.g. 'X (25g)', 'Y (25g)', 'Z (25g)').

    Returns:
        data_x, data_y, data_z, fs
    """
    path = Path(path)

    # Open the IDE file
    doc = get_doc(str(path))

    # Get channel table (Styler or DataFrame) and normalize to DataFrame
    chan_table_raw = get_channel_table(doc)
    chan_df = chan_table_raw.data if hasattr(chan_table_raw, "data") else chan_table_raw

    # Basic sanity check
    for col in ("units", "samples", "rate", "channel", "name"):
        if col not in chan_df.columns:
            raise ValueError(
                f"Channel table for {path} is missing expected column '{col}'. "
                f"Columns are: {list(chan_df.columns)}"
            )

    # Filter to acceleration channels by units='g'
    accel_rows = chan_df[chan_df["units"] == "g"].copy()
    if accel_rows.empty:
        raise ValueError(
            f"No acceleration channels (units='g') found in file {path}. "
            f"Columns: {list(chan_df.columns)}"
        )

    # Keep only the accel rows with the highest sample count
    max_samples = accel_rows["samples"].max()
    accel_rows = accel_rows[accel_rows["samples"] == max_samples].copy()

    # Get the parent Channel of the first accel subchannel
    first_subchannel = accel_rows["channel"].iloc[0]
    parent_channel = first_subchannel.parent

    # Convert that parent channel to a pandas DataFrame
    # Columns will correspond to subchannel names (e.g. 'X (25g)', 'Y (25g)', etc.)
    df = to_pandas(parent_channel)

    # Names of the accel subchannels we care about, in table order
    axis_names = list(accel_rows["name"])

    # If more than 3 accel subchannels, just take the first 3
    if len(axis_names) > 3:
        axis_names = axis_names[:3]

    # If fewer than 3, we'll reuse the first as needed
    while len(axis_names) < 3:
        axis_names.append(axis_names[0])

    # Extract NumPy arrays for three axes
    data_x = df[axis_names[0]].to_numpy()
    data_y = df[axis_names[1]].to_numpy()
    data_z = df[axis_names[2]].to_numpy()

    # Sample rate from the table (Hz)
    fs = float(accel_rows["rate"].iloc[0])

    return data_x, data_y, data_z, fs
