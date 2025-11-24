from vibtool.db import load_config, get_connection, initialize_schema

def main():
    cfg = load_config()
    conn = get_connection(cfg["db_path"])
    initialize_schema(conn)
    print("Database schema created / verified.")

if __name__ == "__main__":
    main()
