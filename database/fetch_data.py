import sqlite3

def query_db():
    # Connect to the SQLite database
    conn = sqlite3.connect('C:\military_vehicle_detection\database\file_uploads.db')
    c = conn.cursor()

    # Query all records from the uploads table
    c.execute("SELECT * FROM uploads")

    # Fetch all rows from the query
    rows = c.fetchall()

    # Close the connection to the database
    conn.close()

    # Print the fetched rows
    for row in rows:
        print(row)

query_db()
