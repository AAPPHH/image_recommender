import faiss
import numpy as np
import sqlite3
import pickle


# index = faiss.read_index("index_hnsw_sift_vlad.faiss")
# vecs = index.reconstruct_n(0, 10)
# for i, v in enumerate(vecs):
#     print(f"Vektor {i}: mean={np.mean(v):.4f}, std={np.std(v):.4f}, min={np.min(v):.4f}, max={np.max(v):.4f}")
# import sqlite3, pickle, numpy as np

conn = sqlite3.connect("images.db")
# cur = conn.cursor()
# cur.execute("SELECT sift_vlad_vector_blob FROM images WHERE sift_vlad_vector_blob IS NOT NULL LIMIT 10")
# rows = [pickle.loads(blob) for (blob,) in cur.fetchall()]
# for i, v in enumerate(rows):
#     print(f"DB-Vektor {i}: mean={np.mean(v):.4f}, std={np.std(v):.4f}, min={np.min(v):.4f}, max={np.max(v):.4f}")
# conn.close()
# Hole eine existierende Bild-Pfad aus der DB, das im Index steckt:
# conn = sqlite3.connect("images.db")
# cur = conn.cursor()
# cur.execute("SELECT sift_vlad_vector_blob FROM images WHERE sift_vlad_vector_blob IS NOT NULL LIMIT 2")
# blob1, blob2 = cur.fetchall()
# vec1 = pickle.loads(blob1[0]).astype("float32").ravel()
# vec2 = pickle.loads(blob2[0]).astype("float32").ravel()
# conn.close()
# print("Manuelle Distanz:", np.linalg.norm(vec1 - vec2))

# conn = sqlite3.connect("images.db")
cur = conn.cursor()
cur.execute("SELECT sift_vlad_vector_blob FROM images WHERE sift_vlad_vector_blob IS NOT NULL LIMIT 100")
blobs = cur.fetchall()
vecs = np.stack([pickle.loads(blob[0]).astype("float32").ravel() for blob in blobs])
conn.close()

# 2. FLAT FAISS Index
index_flat = faiss.IndexFlatL2(vecs.shape[1])
index_flat.add(vecs)

# 3. Nehme den ersten Vektor als Query
dists, idxs = index_flat.search(vecs[:1], 5)
print("FlatIndex Distanzen:", dists)


index = faiss.IndexHNSWFlat(vecs.shape[1], 32)     # M=32, wie im großen Index
index.hnsw.efConstruction = 200                    # gleiche Parameter wie im Build
index.hnsw.efSearch = 50
index.add(vecs)

# Query: Nimm den ersten Vektor (der steckt im Index!)
dists, idxs = index.search(vecs[:1], 5)
print("HNSW Distanzen:", dists)
print("HNSW Indices:", idxs)
