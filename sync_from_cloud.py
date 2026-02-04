import sqlite3
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
# Use the URL from your environment, or fallback to the one you provided
# (Ideally, keep this in your .env file only!)
CLOUD_DB_URL = os.getenv("DATABASE_URL")
LOCAL_DB_FILE = "fyp_database.db"

# List of tables you want to sync (Order matters due to foreign keys!)
TABLES_TO_SYNC = [
    "lecturers",
    "modules",
    "students",
    "grades",
    "risk_scores",
    "risk_history",
    "notifications"
]


def get_sqlite_type(pg_type):
    """Maps PostgreSQL data types to SQLite types."""
    pg_type = pg_type.lower()
    if "int" in pg_type:
        return "INTEGER"
    elif "char" in pg_type or "text" in pg_type:
        return "TEXT"
    elif "real" in pg_type or "double" in pg_type or "numeric" in pg_type or "float" in pg_type:
        return "REAL"
    elif "bool" in pg_type:
        return "INTEGER"  # SQLite uses 0/1 for booleans
    elif "timestamp" in pg_type or "date" in pg_type:
        return "TEXT"
    return "TEXT"  # Fallback


def sync_database():
    print(f"🔌 Connecting to Cloud DB...")
    try:
        pg_conn = psycopg2.connect(CLOUD_DB_URL)
        pg_cursor = pg_conn.cursor()
    except Exception as e:
        print(f"❌ Failed to connect to Cloud DB: {e}")
        return

    print(f"🔌 Connecting to Local SQLite DB ({LOCAL_DB_FILE})...")
    sl_conn = sqlite3.connect(LOCAL_DB_FILE)
    sl_cursor = sl_conn.cursor()

    # Enable foreign keys in SQLite
    sl_cursor.execute("PRAGMA foreign_keys = OFF;")

    for table in TABLES_TO_SYNC:
        print(f"\n--- Syncing Table: {table} ---")

        # 1. Get Schema from Postgres
        try:
            pg_cursor.execute(f"""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = '{table}' 
                ORDER BY ordinal_position;
            """)
            columns = pg_cursor.fetchall()

            if not columns:
                print(f"⚠️ Table '{table}' not found in Cloud DB. Skipping.")
                continue

            # 2. Build CREATE TABLE statement for SQLite
            col_defs = []
            col_names = []
            for col_name, data_type in columns:
                sqlite_type = get_sqlite_type(data_type)
                col_defs.append(f"{col_name} {sqlite_type}")
                col_names.append(col_name)

            # Simple primary key assumption: usually first column or 'id'
            # For this script, we won't strictly enforce PK/FK constraints in creation
            # to avoid complexity, but data will be consistent.
            create_query = f"CREATE TABLE {table} ({', '.join(col_defs)});"

            # 3. Re-create Local Table
            sl_cursor.execute(f"DROP TABLE IF EXISTS {table};")
            sl_cursor.execute(create_query)
            print(f"✅ Re-created structure for '{table}'")

            # 4. Fetch Data from Postgres
            pg_cursor.execute(f"SELECT * FROM {table}")
            rows = pg_cursor.fetchall()

            # 5. Insert Data into SQLite
            if rows:
                placeholders = ", ".join(["?"] * len(col_names))
                insert_query = f"INSERT INTO {table} VALUES ({placeholders})"
                sl_cursor.executemany(insert_query, rows)
                print(f"✅ Imported {len(rows)} rows.")
            else:
                print("ℹ️ Table is empty.")

        except Exception as e:
            print(f"❌ Error syncing {table}: {e}")

    # Re-enable foreign keys
    sl_cursor.execute("PRAGMA foreign_keys = ON;")

    sl_conn.commit()
    sl_conn.close()
    pg_conn.close()
    print("\n✨ Sync Complete! Your local database is now identical to the Cloud DB.")


if __name__ == "__main__":
    sync_database()