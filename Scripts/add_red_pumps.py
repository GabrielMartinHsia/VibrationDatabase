from vibtool.db import load_config, get_connection


def get_or_create_pump(conn, site: str, name: str, tag: str | None, description: str | None) -> int:
    cur = conn.cursor()

    # Check if pump already exists
    cur.execute(
        "SELECT pump_id FROM pump WHERE site = ? AND name = ?",
        (site, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Insert new pump
    cur.execute(
        "INSERT INTO pump (site, name, tag, description) VALUES (?, ?, ?, ?)",
        (site, name, tag, description),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_location(
    conn,
    pump_id: int,
    name: str,
    description: str | None,
    x_dir: str | None,
    y_dir: str | None,
    z_dir: str | None,
) -> int:
    cur = conn.cursor()

    # Check if location already exists
    cur.execute(
        "SELECT location_id FROM measurement_location WHERE pump_id = ? AND name = ?",
        (pump_id, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Insert new location with orientation
    cur.execute(
        """
        INSERT INTO measurement_location (pump_id, name, description, x_dir, y_dir, z_dir)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (pump_id, name, description, x_dir, y_dir, z_dir),
    )
    conn.commit()
    return cur.lastrowid



def main():
    cfg = load_config()
    conn = get_connection(cfg["db_path"])

    pumps = {
        "P34A": "Red site injection pump A",
        "P34B": "Red site injection pump B",
    }

    # Same 9 locations for both pumps
    locations = {
        "L01": "Motor Outboard Bearing",
        "L02": "Motor Inboard Bearing",
        "L03": "Skid Plate at MIB",
        "L04": "Foundation",
        "L05": "Thrust Chamber",
        "L06": "Barrel Suction End",
        "L07": "Barrel Suction Mid",
        "L08": "Barrel Discharge Mid",
        "L09": "Barrel Discharge End",
    }

    orientations = {
        "L01": {"X": "vertical", "Y": "horizontal",   "Z": "axial"},
        "L02": {"X": "vertical", "Y": "horizontal",   "Z": "axial"},
        "L03": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
        "L04": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
        "L05": {"X": "vertical", "Y": "axial",   "Z": "horizontal"},
        "L06": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
        "L07": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
        "L08": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
        "L09": {"X": "axial", "Y": "horizontal",   "Z": "vertical"},
    }


    for pump_name, pump_desc in pumps.items():
        pump_id = get_or_create_pump(
            conn,
            site="Red",
            name=pump_name,
            tag=pump_name,
            description=pump_desc,
        )
        print(f"Pump {pump_name} → pump_id {pump_id}")

        for loc_name, loc_desc in locations.items():
            orient = orientations[loc_name]
            loc_id = get_or_create_location(
                conn,
                pump_id,
                loc_name,
                loc_desc,
                orient["X"],
                orient["Y"],
                orient["Z"],
            )
            print(
                f"  Location {loc_name} ({loc_desc}) "
                f"X={orient['X']} Y={orient['Y']} Z={orient['Z']} → location_id {loc_id}"
            )



if __name__ == "__main__":
    main()
