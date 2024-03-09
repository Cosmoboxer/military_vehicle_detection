import sqlite3

def init_db():
    conn = sqlite3.connect('file_uploads.db')
    c = conn.cursor()
    # Create table
    c.execute('''CREATE TABLE IF NOT EXISTS uploads
                 (filename text, details text)''')
    conn.commit()
    conn.close()

init_db()
