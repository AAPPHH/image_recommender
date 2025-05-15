import os
import logging
import sqlite3
import pickle
import numpy as np
import torch
import faiss
import cv2
from pathlib import Path
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import lpips
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from dreamsim import dreamsim

# ----------------------------------------
# Configuration
# ----------------------------------------
TOP_K = 5
USE_GPU = True
# Basis-Verzeichnis für Deine Bilder (images_v3)
BASE_DIR = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"
# Pfad zur SQLite-DB
DB_PATH = "images.db"

# Logging einstellen
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Device festlegen
device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# ----------------------------------------
# Lazy-Loading für Modelle (LPIPS, DreamSim)
# ----------------------------------------
_lpips_model = None
_ds_model = None
_ds_preprocess = None

def _load_lpips():
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex").to(device).half().eval()
    return _lpips_model


def _load_dreamsim():
    global _ds_model, _ds_preprocess
    if _ds_model is None or _ds_preprocess is None:
        _ds_model, _ds_preprocess = dreamsim(
            pretrained=True,
            device=device,
            normalize_embeds=True,
            dreamsim_type="open_clip_vitb32"
        )
        _ds_model.eval()
    return _ds_model, _ds_preprocess

# ----------------------------------------
# DB-Funktionen: Vektoren lesen/schreiben
# ----------------------------------------
def _get_db_vector(path_rel: str, column: str) -> np.ndarray:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT {column} FROM images WHERE path = ?", (path_rel,))
    row = cur.fetchone()
    conn.close()
    if row and row[0] is not None:
        blob = row[0]
        # Versuche Pickle-Unpacking (ältere Pickle-Blobs)
        try:
            arr = pickle.loads(blob)
            if isinstance(arr, np.ndarray):
                # Logge, dass der Vektor genutzt wurde
                logging.info(f"Loaded '{column}' from DB for '{path_rel}' (shape: {arr.shape}) via pickle.")
                return arr.reshape(1, -1) if arr.ndim == 1 else arr
        except Exception:
            pass
        # Fallback auf rohe Float32-Bytes
        size = len(blob)
        fmt_size = np.dtype('float32').itemsize
        if size % fmt_size != 0:
            logging.warning(
                f"Ungültige Blob-Größe {size} Byte für '{column}' beim Pfad '{path_rel}', überspringe Cache."
            )
            return None
        arr = np.frombuffer(blob, dtype="float32").reshape(1, -1)
        logging.info(f"Loaded '{column}' from DB for '{path_rel}' (shape: {arr.shape}) via raw bytes.")
        return arr
    return None


def _set_db_vector(path_rel: str, column: str, vec: np.ndarray):
    # Speichere immer als Pickle für Kompatibilität
    blob = pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO images (path) VALUES (?)", (path_rel,))
    cur.execute(f"UPDATE images SET {column} = ? WHERE path = ?", (blob, path_rel))
    conn.commit()
    conn.close()
    logging.info(f"Saved '{column}' to DB for '{path_rel}' (shape: {vec.shape}).")

# ----------------------------------------
# Feature-Extraktion mit DB-Cache
# ----------------------------------------
def extract_color_features(image: Image.Image, path_rel: str, bins: int = 16) -> np.ndarray:
    col = "color_vector_blob"
    exists = _get_db_vector(path_rel, col)
    if exists is not None:
        return exists
    logging.info(f"Computing color features for '{path_rel}'.")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([cv_img], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.flatten())
    vec = np.concatenate(feats).astype("float32").reshape(1, -1)
    _set_db_vector(path_rel, col, vec)
    return vec


def extract_lpips_features(image: Image.Image, path_rel: str) -> np.ndarray:
    col = "lpips_vector_blob"
    exists = _get_db_vector(path_rel, col)
    if exists is not None:
        return exists
    logging.info(f"Computing LPIPS features for '{path_rel}'.")
    model = _load_lpips()
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Lambda(lambda img: (to_tensor(img).float() - 0.5) / 0.5),
    ])
    x = transform(image).unsqueeze(0).to(device).half()
    with torch.no_grad():
        feats = model.net(x)
    fmap = feats[-1] if isinstance(feats, (list, tuple)) else feats
    vec = torch.nn.functional.normalize(
        torch.nn.functional.adaptive_avg_pool2d(fmap, 1).flatten(1), dim=1
    )
    arr = vec.cpu().numpy().astype("float32")
    _set_db_vector(path_rel, col, arr)
    return arr


def extract_dreamsim_features(image: Image.Image, path_rel: str) -> np.ndarray:
    col = "dreamsim_vector_blob"
    exists = _get_db_vector(path_rel, col)
    if exists is not None:
        return exists
    logging.info(f"Computing DreamSim features for '{path_rel}'.")
    model, preprocess = _load_dreamsim()
    tensor = preprocess(image)
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    x = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.embed(x)
    arr = vec.cpu().numpy().astype("float32")
    _set_db_vector(path_rel, col, arr)
    return arr

# ----------------------------------------
# Suche ähnliche Bilder mit FAISS
# ----------------------------------------
def search_similar_images(query_image_path: str, index_type: str = "clip"):
    try:
        image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Fehler beim Öffnen des Abfragebildes: {e}")
        return

    path_rel = os.path.relpath(query_image_path, BASE_DIR)
    requested = set(index_type.lower().split("_"))
    valid = ["clip", "color", "lpips", "dreamsim"]
    ordered = [v for v in valid if v in requested]
    if not ordered:
        logging.error(f"Unbekannter index_type '{index_type}'. Wähle aus {valid}.")
        return
    logging.info(f"🔍 Extrahiere Features in Reihenfolge: {ordered}")

    parts = []
    for vec_type in ordered:
        if vec_type == "color":
            parts.append(extract_color_features(image, path_rel))
        elif vec_type == "lpips":
            parts.append(extract_lpips_features(image, path_rel))
        elif vec_type in ("dreamsim", "clip"):
            parts.append(extract_dreamsim_features(image, path_rel))

    query_vec = np.concatenate(parts, axis=1).astype("float32")
    canonical = "_".join(ordered)
    index_file = f"index_hnsw_{canonical}.faiss"
    offset_table = f"faiss_index_offsets_{canonical}"

    try:
        index = faiss.read_index(index_file)
        logging.info(f"Geladener FAISS-Index '{index_file}' mit {index.ntotal} Vektoren.")
    except Exception as e:
        logging.error(f"Fehler beim Laden des Index '{index_file}': {e}")
        return

    distances, indices = index.search(query_vec, TOP_K)
    logging.info(f"Distanzen: {distances[0]}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for rank, offset in enumerate(indices[0]):
        cur.execute(
            f"SELECT image_id FROM {offset_table} WHERE offset = ?", (int(offset),)
        )
        row = cur.fetchone()
        if not row:
            logging.warning(f"Kein Eintrag für offset={offset} in {offset_table}")
            continue
        img_id = row[0]
        cur.execute("SELECT path FROM images WHERE id = ?", (img_id,))
        fp_row = cur.fetchone()
        if not fp_row:
            logging.warning(f"Kein Pfad für id={img_id}")
            continue
        full_path = os.path.join(BASE_DIR, fp_row[0])
        results.append((full_path, float(distances[0, rank])))
    conn.close()

    if not results:
        logging.error("Keine ähnlichen Bilder gefunden.")
        return
    results.sort(key=lambda x: x[1])

    sns.set_theme(style="whitegrid")
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (fp, dist) in zip(axes, results):
        try:
            img = Image.open(fp)
            ax.imshow(img)
            ax.set_title(f"{Path(fp).name}\nDist: {dist:.4f}")
            ax.axis("off")
        except Exception as e:
            logging.error(f"Fehler beim Rendern von {fp}: {e}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query_image = (
        r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones"
        r"\image_recomender\image_recommender\images_v3"
        r"\image_data\pixabay_dataset_v1\images_01"
        r"\a-heart-love-sadness-emotions-2719081.jpg"
    )
    search_similar_images(query_image, index_type="dreamsim")
