SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pump (
    pump_id       INTEGER PRIMARY KEY,
    site          TEXT NOT NULL,
    name          TEXT NOT NULL,
    tag           TEXT,
    description   TEXT
);

CREATE TABLE IF NOT EXISTS measurement_location (
    location_id   INTEGER PRIMARY KEY,
    pump_id       INTEGER NOT NULL,
    name          TEXT NOT NULL,
    description   TEXT,
    x_dir         TEXT,   -- e.g. 'horizontal', 'vertical', 'axial'
    y_dir         TEXT,
    z_dir         TEXT,
    axis_notes    TEXT,
    FOREIGN KEY (pump_id) REFERENCES pump(pump_id)
);

CREATE TABLE IF NOT EXISTS measurement_run (
    reading_id          INTEGER PRIMARY KEY,
    location_id         INTEGER NOT NULL,
    timestamp_utc       TEXT NOT NULL,
    operating_state     TEXT,
    sample_rate_hz      REAL,
    duration_s          REAL,
    ide_file_path       TEXT NOT NULL,
    notes               TEXT,
    FOREIGN KEY (location_id) REFERENCES measurement_location(location_id)
);

CREATE TABLE IF NOT EXISTS fft_spectrum (
    reading_id       INTEGER NOT NULL,
    axis             TEXT NOT NULL CHECK (axis IN ('X','Y','Z')),
    nfft             INTEGER NOT NULL,
    sample_rate_hz   REAL NOT NULL,
    window           TEXT,
    spectrum_type    TEXT NOT NULL,
    spectrum_blob    BLOB NOT NULL,
    PRIMARY KEY (reading_id, axis),
    FOREIGN KEY (reading_id) REFERENCES measurement_run(reading_id)
);

CREATE TABLE IF NOT EXISTS band_level (
    reading_id      INTEGER NOT NULL,
    axis        TEXT NOT NULL CHECK (axis IN ('X','Y','Z')),
    f_low_hz    REAL NOT NULL,
    f_high_hz   REAL NOT NULL,
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    PRIMARY KEY (reading_id, axis, f_low_hz, f_high_hz, metric),
    FOREIGN KEY (reading_id) REFERENCES measurement_run(reading_id)
);
"""
