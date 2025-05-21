import sqlite3

# conn = sqlite3.connect("images.db")
# cur = conn.cursor()
# cur.execute("ALTER TABLE images DROP COLUMN sift_vlad_blob")
# conn.commit()
# conn.close()
# conn = sqlite3.connect("images.db")
# cur = conn.cursor()
# cur.execute("PRAGMA table_info(images)")
# columns = [row[1] for row in cur.fetchall()]
# print("Spalten im Table 'images':", columns)
# conn.close()


# conn = sqlite3.connect("images.db")
# cur = conn.cursor()

# # Spalten vor dem Umbenennen
# cur.execute("PRAGMA table_info(images)")
# print("Vorher:", [row[1] for row in cur.fetchall()])

# # Umbenennen
# cur.execute("ALTER TABLE images RENAME COLUMN sift_vlad_blob TO sift_vlad_vector_blob;")
# conn.commit()

# # Spalten nach dem Umbenennen
# cur.execute("PRAGMA table_info(images)")
# print("Nachher:", [row[1] for row in cur.fetchall()])

# conn.close()


# import sqlite3
# import pickle
# import numpy as np

# conn = sqlite3.connect("images.db")
# cur = conn.cursor()
# cur.execute("SELECT sift_vlad_vector_blob FROM images WHERE sift_vlad_vector_blob IS NOT NULL LIMIT 10")
# rows = cur.fetchall()
# for i, (blob,) in enumerate(rows):
#     try:
#         vec = pickle.loads(blob)
#     except Exception:
#         vec = np.frombuffer(blob, dtype="float32")
#     print(f"Sample {i}: mean={np.mean(vec):.4f} std={np.std(vec):.4f} min={np.min(vec):.4f} max={np.max(vec):.4f}")
# conn.close()
import sqlite3
import pickle
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

conn = sqlite3.connect("images.db")
cur = conn.cursor()
cur.execute("SELECT sift_vlad_vector_blob FROM images WHERE sift_vlad_vector_blob IS NOT NULL LIMIT 100")
rows = [pickle.loads(blob) for (blob,) in cur.fetchall()]
conn.close()

X = np.stack(rows)
D = squareform(pdist(X, metric="euclidean"))
plt.imshow(D, cmap="viridis")
plt.colorbar(label="Distanz")
plt.title("Paarweise Distanzen (SIFT-VLAD-PCA)")
plt.show()

print("Min:", D.min(), "Max:", D.max(), "Median:", np.median(D))
