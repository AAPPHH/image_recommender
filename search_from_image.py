import os
import logging
import sqlite3
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from dreamsim import dreamsim
import joblib

from vector_scripts.creat_vector_base import BaseVectorIndexer, load_image

class ImageRecommender:
    def __init__(
        self,
        db_path="images.db",
        use_gpu=True,
        sift_codebook_path="sift_codebook.npy",
        sift_pca_path="sift_vlad_pca.joblib",
        sift_n_clusters=256,
        sift_desc_dim=128,
        top_k=5
    ):
        self.base_dir = Path().expanduser().resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.sift_codebook_path = Path(sift_codebook_path).expanduser().resolve()
        self.sift_pca_path = Path(sift_pca_path).expanduser().resolve()
        self.sift_n_clusters = sift_n_clusters
        self.sift_desc_dim = sift_desc_dim
        self.top_k = top_k

        self._ds_model = None
        self._ds_preprocess = None
        self._sift_codebook = None
        self._sift_pca = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    # ----------- Model/Component Loading ----------- #
    def _load_dreamsim(self):
        if self._ds_model is None or self._ds_preprocess is None:
            self._ds_model, self._ds_preprocess = dreamsim(
                pretrained=True,
                device=self.device,
                normalize_embeds=True,
                dreamsim_type="ensemble",
            )
            self._ds_model.eval()
        return self._ds_model, self._ds_preprocess

    def _load_sift_components(self):
        if self._sift_codebook is None:
            if not os.path.exists(self.sift_codebook_path):
                raise FileNotFoundError(
                    f"Codebook fehlt: {self.sift_codebook_path}. "
                    "Bitte zuerst mit dem Indexer erzeugen."
                )
            self._sift_codebook = np.load(self.sift_codebook_path).astype("float32")
        if self._sift_pca is None:
            if not os.path.exists(self.sift_pca_path):
                raise FileNotFoundError(
                    f"PCA fehlt: {self.sift_pca_path}. "
                    "Bitte zuerst mit dem Indexer erzeugen."
                )
            self._sift_pca = joblib.load(self.sift_pca_path)
        return self._sift_codebook, self._sift_pca

    # ----------- DB Vector Reading ----------- #
    def _get_db_vector(self, path_rel: str, vector_table: str, vector_column: str) -> np.ndarray:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT id FROM images WHERE path = ?", (path_rel,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return None
        image_id = row[0]
        cur.execute(f"SELECT {vector_column} FROM {vector_table} WHERE image_id = ?", (image_id,))
        vrow = cur.fetchone()
        conn.close()
        if vrow and vrow[0] is not None:
            blob = vrow[0]
            try:
                arr = pickle.loads(blob)
                if isinstance(arr, np.ndarray):
                    logging.info(f"Loaded '{vector_column}' from DB for '{path_rel}' (shape: {arr.shape}) via pickle.")
                    return arr.reshape(1, -1) if arr.ndim == 1 else arr
            except Exception:
                pass
            size = len(blob)
            fmt_size = np.dtype('float32').itemsize
            if size % fmt_size != 0:
                logging.warning(
                    f"Ungültige Blob-Größe {size} Byte für '{vector_column}' beim Pfad '{path_rel}', überspringe Cache."
                )
                return None
            arr = np.frombuffer(blob, dtype="float32").reshape(1, -1)
            logging.info(f"Loaded '{vector_column}' from DB for '{path_rel}' (shape: {arr.shape}) via raw bytes.")
            return arr
        return None

    # ----------- Feature Extraction ----------- #
    def extract_color_features(self, img_path: str, path_rel: str, bins: int = 16) -> np.ndarray:
        col = "color_vector_blob"
        table = "color_vectors"
        exists = self._get_db_vector(path_rel, table, col)
        if exists is not None:
            return exists
        logging.info(f"Computing color features for '{path_rel}' via ColorVectorIndexer.")
        from vector_scripts.create_color_vector import ColorVectorIndexer 
        load_image_kwargs = dict(use_cv2=True)
        images_dir = self.base_dir
        args = (path_rel, images_dir, bins, load_image_kwargs)
        vec = ColorVectorIndexer.compute_color_vector_worker(args)
        if vec is None:
            logging.error(f"Failed to compute color vector for '{img_path}'.")
            return None
        return vec.reshape(1, -1)

    def extract_sift_vlad_features(self, img_path: str, path_rel: str) -> np.ndarray:
        col = "sift_vector_blob"
        table = "sift_vectors"
        cached = self._get_db_vector(path_rel, table, col)
        if cached is not None:
            return cached
        logging.warning(f"SIFT-VLAD nicht im Cache – berechne on-the-fly für '{path_rel}' via SIFTVLADVectorIndexer.")
        from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
        # Codebook und ggf. PCA laden wie gehabt (bzw. aus Indexer holen)
        codebook, pca = self._load_sift_components()
        images_dir = self.base_dir
        img_size = (512, 512)  # Standard für SIFT-VLAD bei dir
        args = (path_rel, images_dir, codebook, img_size)
        vlad = SIFTVLADVectorIndexer.compute_sift_vlad_vector_worker(args)
        if vlad is None:
            logging.error(f"Failed to compute SIFT-VLAD vector for '{img_path}'.")
            return None
        # Optional: Encoder oder PCA (wie in deinem Batch-Indexer)
        if pca is not None:
            vlad = pca.transform(vlad.reshape(1, -1)).astype(np.float32)
            vlad /= np.linalg.norm(vlad)
        else:
            vlad = vlad.reshape(1, -1)
        return vlad

    def extract_dreamsim_features(self, img_path: str, path_rel: str) -> np.ndarray:
        table = "dreamsim_vectors"
        column = "dreamsim_vector_blob"
        exists = self._get_db_vector(path_rel, table, column)
        if exists is not None:
            return exists
        logging.info(f"Computing DreamSim features for '{path_rel}' via DreamSimVectorIndexer.")
        from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer
        # Modell und Preprocess ggf. lazy laden
        model, preprocess = self._load_dreamsim()
        images_dir = self.base_dir
        args = (path_rel, images_dir, preprocess, model, self.device)
        vec = DreamSimVectorIndexer.compute_dreamsim_vector_worker(args)
        if vec is None:
            logging.error(f"Failed to compute DreamSim vector for '{img_path}'.")
            return None
        return vec


    # ----------- Main Search Method ----------- #
    def search_similar_images(self, query_image_path: str, index_type: str = "color"):
        path_rel = os.path.relpath(query_image_path, self.base_dir)
        requested = [x.strip() for x in index_type.lower().split(",")]
        valid = ["color", "hog", "lpips", "dreamsim", "sift", "color_sift", "sift_dreamsim"]
        ordered = [v for v in valid if v in requested]
        if not ordered:
            logging.error(f"Unbekannter index_type '{index_type}'. Wähle aus {valid}.")
            return
        logging.info(f"🔍 Extrahiere Features in Reihenfolge: {ordered}")
        parts = []
        for vec_type in ordered:
            if vec_type == "color":
                parts.append(self.extract_color_features(query_image_path, path_rel))
            elif vec_type == "sift":
                parts.append(self.extract_sift_vlad_features(query_image_path, path_rel))
            elif vec_type == "dreamsim":
                parts.append(self.extract_dreamsim_features(query_image_path, path_rel))
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
        distances, indices = index.search(query_vec, self.top_k)
        logging.info(f"Distanzen: {distances[0]}")
        conn = sqlite3.connect(self.db_path)
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
            full_path = os.path.join(self.base_dir, "images_v3", fp_row[0])
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
        img = load_image(query_image_path)
        axes[0].imshow(img)
        axes[0].set_title("Query Image")
        axes[0].axis("off")
        for idx, (fp, dist) in enumerate(results):
            ax = axes[idx + 1]
            try:
                img = load_image(fp)
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
        r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3\image_data\weather_image_recognition\rainbow\0594.jpg"
    )
    rec = ImageRecommender()
    rec.search_similar_images(query_image, index_type="color")