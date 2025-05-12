import os
import torch
import sqlite3
import faiss
from PIL import Image
import logging
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import lpips
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor
from pathlib import Path
from dreamsim import dreamsim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TOP_K = 5
USE_GPU = True
BASE_DIR = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recommender\image_recommender\images_v3"
DB_PATH = "images.db"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")

# --- Load models ---
# LPIPS perceptual-similarity model
lpips_model = lpips.LPIPS(net="alex").to(device).half().eval()

# DreamSim (CLIP-based) embedding model
ds_model, ds_preprocess = dreamsim(
    pretrained=True,
    device=device,
    normalize_embeds=True,
    dreamsim_type="open_clip_vitb32"
)
ds_model.eval()

def extract_color_features(image: Image.Image, bins: int = 16) -> np.ndarray:
    """Compute normalized per-channel histograms concatenated into one vector."""
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([cv_img], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.flatten())
    return np.concatenate(feats).astype("float32").reshape(1, -1)

def extract_lpips_features(image: Image.Image) -> np.ndarray:
    """Extract LPIPS perceptual features."""
    transform = T.Compose([
        T.Resize((224, 224), antialias=True),
        T.Lambda(lambda img: (to_tensor(img).float() - 0.5) / 0.5),
    ])
    x = transform(image).unsqueeze(0).to(device).half()
    with torch.no_grad():
        feats = lpips_model.net(x)
    fmap = feats[-1] if isinstance(feats, (list, tuple)) else feats
    vec = torch.nn.functional.normalize(
        torch.nn.functional.adaptive_avg_pool2d(fmap, 1).flatten(1),
        dim=1
    )
    return vec.cpu().numpy().astype("float32")

def extract_dreamsim_features(image: Image.Image) -> np.ndarray:
    """Extract DreamSim/CLIP-based embedding."""
    tensor = ds_preprocess(image)
    if tensor.ndim == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    x = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        vec = ds_model.embed(x)
    return vec.cpu().numpy().astype("float32")

def search_similar_images(query_image_path: str, index_type: str = "clip"):
    """Search for TOP_K most similar images using a FAISS index."""
    # Load the query image
    try:
        image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening query image: {e}")
        return

    requested = set(index_type.lower().split("_"))
    valid = ["clip", "color", "lpips", "dreamsim"]
    ordered = [v for v in valid if v in requested]
    if not ordered:
        logging.error(f"Unknown index_type '{index_type}'. Choose from {valid}.")
        return

    logging.info(f"🔍 Extracting features in order: {ordered}")

    parts = []
    for vec_type in ordered:
        if vec_type == "color":
            parts.append(extract_color_features(image))
        elif vec_type == "lpips":
            parts.append(extract_lpips_features(image))
        elif vec_type == "dreamsim":
            parts.append(extract_dreamsim_features(image))
    query_vec = np.concatenate(parts, axis=1).astype("float32")

    canonical = "_".join(ordered)
    index_file = f"index_hnsw_{canonical}.faiss"
    offset_table = f"faiss_index_offsets_{canonical}"

    try:
        index = faiss.read_index(index_file)
        logging.info(f"Loaded FAISS index '{index_file}' ({index.ntotal} vectors).")
    except Exception as e:
        logging.error(f"Error loading index '{index_file}': {e}")
        return

    distances, indices = index.search(query_vec, TOP_K)
    logging.info(f"Distances: {distances[0]}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    results = []
    for rank, offset in enumerate(indices[0]):
        cur.execute(
            f"SELECT image_id FROM {offset_table} WHERE offset = ?",
            (int(offset),)
        )
        row = cur.fetchone()
        if not row:
            logging.warning(f"No entry for offset={offset} in {offset_table}")
            continue
        img_id = row[0]
        cur.execute("SELECT path FROM images WHERE id = ?", (img_id,))
        fp_row = cur.fetchone()
        if not fp_row:
            logging.warning(f"No path for id={img_id}")
            continue
        full_path = os.path.join(BASE_DIR, fp_row[0])
        results.append((full_path, float(distances[0, rank])))
    conn.close()

    if not results:
        logging.error("No similar images found.")
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
            ax.set_title(f"{Path(fp).name}\n{dist:.4f}")
            ax.axis("off")
        except Exception as e:
            logging.error(f"Error rendering {fp}: {e}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query_image = (
        r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones"
        r"\image_recommender\image_recommender\images_v3"
        r"\image_data\pixabay_dataset_v1\images_01"
        r"\a-heart-love-sadness-emotions-2719081.jpg"
    )

    # Example: search using LPIPS and DreamSim jointly
    search_similar_images(query_image, index_type="lpips_dreamsim")
