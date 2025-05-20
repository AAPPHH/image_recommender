import sqlite3

conn = sqlite3.connect("images.db")
cur = conn.cursor()
cur.execute("ALTER TABLE images DROP COLUMN sift_vlad_blob")
conn.commit()
conn.close()
