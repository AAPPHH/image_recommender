import os
import torch
import open_clip
import sqlite3
import faiss
from PIL import Image
import logging
import seaborn as sns
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- CONFIG ---
TOP_K = 5
USE_GPU = True
BASE_DIR = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class SearchFromImage:
    def __init__(self, device=None):
        if device is None:
            device = torch.device(
                "cuda" if USE_GPU and torch.cuda.is_available() else "cpu"
            )
        self.device = device
        self._setup_model()

    def _setup_model(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-L-14",
            pretrained="openai",
            precision="fp16",
            device=self.device,
            force_quick_gelu=True,
            jit=True,
        )
        self.model.eval()

    def extract_features(self, image):
        processed = self.preprocess(image).unsqueeze(0).to(self.device).half()
        with torch.no_grad():
            features = self.model.encode_image(processed, normalize=True)
        return features.cpu().numpy().astype("float32")


def search_similar_images(query_image_path):
    searcher = SearchFromImage()

    try:
        image = Image.open(query_image_path).convert("RGB")
    except Exception as e:
        logging.error(f"Error opening the image: {e}")


    logging.info(f"🔍 Searching for similar images to: {query_image_path}")
    query_vec = searcher.extract_features(image)

    try:
        index = faiss.read_index("index_hnsw.faiss")
        logging.info("FAISS index loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading the FAISS index: {e}")

    query_vec = query_vec.astype("float32").reshape(1, -1)
    D, I = index.search(query_vec, TOP_K)
    logging.info(f"FAISS search completed. Distances: {D}")

    conn = sqlite3.connect("images.db")
    cursor = conn.cursor()

    results = []
    for i, offset in enumerate(I[0]):
        logging.info(f"[DEBUG] Bearbeite Index {i} mit Offset: {offset}")

        cursor.execute(
            "SELECT filepath FROM images WHERE faiss_index_offset=?", (int(offset),)
        )
        result = cursor.fetchone()
        logging.info(f"[DEBUG] SQL-Ergebnis für Offset {offset}: {result}")

        if result:
            relative_path = result[0]
            db_filepath = os.path.join(BASE_DIR, relative_path)
            logging.info(f"[DEBUG] Zusammengesetzter Dateipfad: {db_filepath}")
            results.append((db_filepath, D[0, i]))
        else:
            logging.warning(f"No entry found in 'images' for FAISS offset {offset}.")
    conn.close()

    if not results:
        logging.error("No similar images found.")

    results = sorted(results, key=lambda x: x[1])
    sns.set_theme(style="whitegrid")
    num_results = len(results)
    fig, axes = plt.subplots(1, num_results, figsize=(5 * num_results, 5))

    if num_results == 1:
        axes = [axes]

    for ax, (fp, dist) in zip(axes, results):
        try:
            img = Image.open(fp)
            ax.imshow(img)
            ax.set_title(f"{os.path.basename(fp)}\nDist: {dist:.4f}", fontsize=10)
            ax.axis("off")
        except Exception as e:
            logging.error(f"Error loading {fp}: {e}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    query_image_path = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3\image_data\DAISY_2025\IMG_3663.jpeg"
    search_similar_images(query_image_path)