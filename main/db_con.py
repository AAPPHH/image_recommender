import sqlite3
import pickle
import numpy as np

def get_random_sift_vectors(db_path, num_samples=10):
    """
    Retrieves a random sample of SIFT vectors from the database.

    Args:
        db_path (str): Path to the SQLite database.
        num_samples (int): Number of random vectors to retrieve.

    Returns:
        List[Tuple[str, np.ndarray or None]]: A list of (image path, vector) pairs.
            If a vector fails to load, its value will be None.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        SELECT i.path, s.sift_vector_blob
        FROM images i
        JOIN sift_vectors s ON i.id = s.image_id
        ORDER BY RANDOM() LIMIT ?
    """, (num_samples,))
    examples = []
    for path, blob in c.fetchall():
        try:
            vec = pickle.loads(blob)
        except Exception as e:
            print(f"Fehler beim Laden von {path}: {e}")
            vec = None
        examples.append((path, vec))
    conn.close()
    return examples

# Beispielnutzung:
DB_PATH = "images.db"
examples = get_random_sift_vectors(DB_PATH, num_samples=10)
for path, vec in examples:
    print(f"{path}: shape={None if vec is None else vec.shape}")

# Statistik:
valid_vecs = [v for p, v in examples if v is not None]
if valid_vecs:
    arr = np.vstack(valid_vecs)
    print(f"Mittelwert: {arr.mean(axis=0)}")
    print(f"Varianz:   {arr.var(axis=0)}")
