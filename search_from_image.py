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
from scipy.spatial.distance import cdist
import joblib

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
            dreamsim_type="ensemble",
        )
        _ds_model.eval()
    return _ds_model, _ds_preprocess

# -------------------------------
# Lazy-Loading: SIFT-VLAD-Komponenten
# -------------------------------
_sift_codebook = None
_sift_pca = None
SIFT_CODEBOOK_PATH = "sift_codebook.npy"
SIFT_PCA_PATH = "sift_vlad_pca.joblib"
SIFT_N_CLUSTERS = 256        # muss zu deinem Codebook passen
SIFT_DESC_DIM  = 128         # SIFT-Descriptor-Länge

def _load_sift_components():
    """
    Lädt Codebook (kMeans-Clusterzentren) und PCA einmalig in den RAM.
    """
    global _sift_codebook, _sift_pca
    if _sift_codebook is None:
        if not os.path.exists(SIFT_CODEBOOK_PATH):
            raise FileNotFoundError(
                f"Codebook fehlt: {SIFT_CODEBOOK_PATH}. "
                "Bitte zuerst mit dem Indexer erzeugen."
            )
        _sift_codebook = np.load(SIFT_CODEBOOK_PATH).astype("float32")

    if _sift_pca is None:
        if not os.path.exists(SIFT_PCA_PATH):
            raise FileNotFoundError(
                f"PCA fehlt: {SIFT_PCA_PATH}. "
                "Bitte zuerst mit dem Indexer erzeugen."
            )
        _sift_pca = joblib.load(SIFT_PCA_PATH)

    return _sift_codebook, _sift_pca


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
    return vec

def extract_sift_vlad_features(image: Image.Image, path_rel: str) -> np.ndarray:
    """
    Liefert den SIFT-VLAD-Vektor (256-D PCA) aus der DB.
    Fällt optional auf On-the-fly-Berechnung zurück,
    falls der Eintrag noch nicht im Cache steht.
    """
    col = "sift_vlad_vector_blob"
    cached = _get_db_vector(path_rel, col)
    if cached is not None:
        return cached

    logging.warning(f"SIFT-VLAD nicht im Cache – berechne on the fly für '{path_rel}'.")
    codebook, pca = _load_sift_components()

    # 1) SIFT-Deskriptoren
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (256, 256))
    sift = cv2.SIFT_create()
    _, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(desc) == 0:
        logging.error(f"Keine SIFT-Features in '{path_rel}'.")
        return None

    # 2) VLAD-Aggregierung
    idxs = np.argmin(cdist(desc, codebook), axis=1)
    vlad = np.zeros((SIFT_N_CLUSTERS, SIFT_DESC_DIM), dtype="float32")
    for i, d in zip(idxs, desc):
        vlad[i] += d - codebook[i]

    # Power- und L2-Norm
    vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
    vlad = vlad.flatten()
    norm = np.linalg.norm(vlad)
    if norm > 0:
        vlad /= norm

    # 3) PCA-Kompression
    vec = pca.transform(vlad.reshape(1, -1)).astype("float32")
    return vec


def extract_hog_features(image: Image.Image, path_rel: str, img_size=(64, 128)) -> np.ndarray:
    col = "hog_vector_blob"
    exists = _get_db_vector(path_rel, col)
    if exists is not None:
        return exists
    logging.info(f"Computing HOG features for '{path_rel}'.")
    img = image.convert("L").resize(img_size)
    img_np = np.array(img)
    hog = cv2.HOGDescriptor()
    vec = hog.compute(img_np)
    if vec is None:
        logging.warning(f"HOG-Feature konnte für '{path_rel}' nicht berechnet werden.")
        return None
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    arr = vec.flatten().astype("float16").reshape(1, -1)
    return arr


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
    return arr

# ----------------------------------------
# Suche ähnliche Bilder mit FAISS
# ----------------------------------------
def search_similar_images(query_image_path: str, index_type: str = "color"):
    try:
        image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Fehler beim Öffnen des Abfragebildes: {e}")
        return

    path_rel = os.path.relpath(query_image_path, BASE_DIR)
    # Akzeptiere nur einen oder mehrere Typen, getrennt durch Komma (nicht Unterstrich)
    requested = [x.strip() for x in index_type.lower().split(",")]
    valid = ["color", "hog", "lpips", "dreamsim", "sift_vlad"]
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
        elif vec_type == "dreamsim":
            parts.append(extract_dreamsim_features(image, path_rel))
        elif vec_type == "hog":
            parts.append(extract_hog_features(image, path_rel))
        elif vec_type == "sift_vlad":
            parts.append(extract_sift_vlad_features(image, path_rel))
        else:
            logging.error(f"Unbekannter Vektor-Typ '{vec_type}' für '{path_rel}'.")
            return


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
    print("==== FAISS DEBUG ====")
    print("QUERY_VEC SHAPE:", query_vec.shape)
    print("QUERY_VEC DTYPE:", query_vec.dtype)
    print("QUERY_VEC MIN/MAX:", np.min(query_vec), np.max(query_vec))
    print("INDEX DIM:", index.d)
    print("INDEX TYPE:", type(index))
    print("=====================")

    distances, indices = index.search(query_vec, TOP_K)
    print("FAISS DISTANCES:", distances)
    print("FAISS INDICES:", indices)

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
    max_per_row = 3
    total_images = len(results) + 1
    ncols = max_per_row
    nrows = int(np.ceil(total_images / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = axes.flatten()

    axes[0].imshow(image)
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for idx, (fp, dist) in enumerate(results):
        ax = axes[idx + 1]
        try:
            img = Image.open(fp)
            ax.imshow(img)
            ax.set_title(f"{Path(fp).name}\nDist: {dist:.4f}")
            ax.axis("off")
        except Exception as e:
            logging.error(f"Fehler beim Rendern von {fp}: {e}")

    for ax in axes[len(results) + 1:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query_image = (
        r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3\image_data\Landscapes\00000000_(5).jpg"
    )
    search_similar_images(query_image, index_type="sift_vlad")
