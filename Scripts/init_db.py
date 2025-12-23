from vibtool.db import load_config, get_connection, initialize_schema, ensure_ide_fingerprint_fields


def main():
    cfg = load_config()
    conn = get_connection(cfg["db_path"])
    initialize_schema(conn)
    ensure_ide_fingerprint_fields(conn)
    print("Database schema created / verified (including IDE fingerprint fields).")


if __name__ == "__main__":
    main()
