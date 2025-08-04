import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import sqlite3
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from vector_scripts.create_vector_base import load_image

class ImageRecommender:
    def __init__(
        self,
        images_root="image_data",
        db_path="images.db",
        use_gpu=True,
        sift_codebook_path="sift_codebook.npy",
        sift_pca_path="sift_vlad_pca.joblib",
        sift_n_clusters=256,
        sift_desc_dim=128,
        top_k=5
    ):
        self.base_dir = Path().expanduser().resolve()
        self.images_root = (self.base_dir / images_root).resolve()
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

    def _get_db_vector(self, path_rel: str, vector_table: str, vector_column: str) -> np.ndarray:
        """
        Retrieves a stored vector from the SQLite database for a given image path.

        Args:
            path_rel (str): Relative image path under the base directory.
            vector_table (str): Name of the feature table (e.g. 'color_vectors').
            vector_column (str): Column name storing the vector blob.

        Returns:
            np.ndarray or None: Deserialized feature vector if available.
        """
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

    def get_or_compute_vector(self, path_rel: str, vector_table: str, vector_column: str, compute_func, reshape: tuple = None, print_vectors: bool = True) -> np.ndarray | None:
        """
        Retrieves a vector from the database if available, otherwise computes it using a provided function.

        Args:
            path_rel (str): Relative path to the image.
            vector_table (str): Table name to query.
            vector_column (str): Column name for the feature blob.
            compute_func (Callable): Function to compute the vector if not cached.
            reshape (tuple, optional): Optional shape to reshape computed vector.
            print_vectors (bool): Whether to print the loaded or computed vector.

        Returns:
            np.ndarray or None: Feature vector or None if computation fails.
        """
        print(f"🔍 Suche Vektor '{vector_column}' für '{path_rel}' in DB...")
        cached = self._get_db_vector(path_rel, vector_table, vector_column)
        if cached is not None:
            print("✅ Vektor aus Cache geladen.")
            if print_vectors:
                print(f"[DB-Vektor]: {cached}")
            return cached

        vec = compute_func()
        if vec is None:
            logging.error(f"Failed to compute {vector_column} for '{path_rel}'.")
            return None
        if print_vectors:
            print(f"[Neu berechneter Vektor]: {vec}")
        if reshape is not None:
            vec = vec.reshape(*reshape)
        return vec

    def extract_color_features(self, path_rel: str) -> np.ndarray:
        """
        Loads or computes the color histogram vector for the given image.

        Args:
            path_rel (str): Relative path to the image.

        Returns:
            np.ndarray: Color feature vector.
        """
        from vector_scripts.create_color_vector import ColorVectorIndexer
        def compute():
            indexer = ColorVectorIndexer(
                db_path="images.db",
                base_dir=self.base_dir,
                log_file="color_indexer.log",
                batch_size=16384
            )
            vec = indexer.compute_vectors([path_rel])[0]
            return vec
        return self.get_or_compute_vector(
            path_rel,
            vector_table="color_vectors",
            vector_column="color_vector_blob",
            compute_func=compute,
            reshape=(1, -1),
        )

    def extract_sift_vlad_features(self, path_rel: str) -> np.ndarray:
        """
        Loads or computes the SIFT-VLAD descriptor vector for the given image.

        Uses a pre-trained visual vocabulary and PCA projection to compress
        dense SIFT features into a compact VLAD vector.

        Args:
            path_rel (str): Relative path to the image.

        Returns:
            np.ndarray: SIFT-VLAD feature vector.
        """
        from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
        def compute():
            indexer = SIFTVLADVectorIndexer(
                db_path="images.db",
                base_dir=self.base_dir, 
                log_file="sift_vlad.log",
                batch_size=256
            )
            return indexer.compute_vectors([path_rel])[0]
        return self.get_or_compute_vector(
            path_rel,
            vector_table="sift_vectors",
            vector_column="sift_vector_blob",
            compute_func=compute,
        )

    def extract_dreamsim_features(self, path_rel: str) -> np.ndarray:
        """
        Loads or computes the DreamSim embedding for the given image.

        DreamSim embeddings are derived from a CLIP-based neural network
        that captures high-level visual semantics for similarity comparison.

        Args:
            path_rel (str): Relative path to the image.

        Returns:
            np.ndarray: DreamSim feature vector.
        """
        from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer
        def compute():
            if not hasattr(self, "_dreamsim_indexer"):
                self._dreamsim_indexer = DreamSimVectorIndexer(
                    db_path=str(self.db_path),
                    base_dir=str(self.base_dir),
                    batch_size=4096,
                    model_batch=128,
                    log_file="dreamsim_indexer.log",
                    log_dir="logs",
                )
            embeddings, valid_paths = self._dreamsim_indexer._batch_image_to_vector([path_rel])
            if len(valid_paths) == 1 and embeddings.shape[0] == 1:
                return embeddings[0].numpy().reshape(1, -1)
        return self.get_or_compute_vector(
            path_rel,
            vector_table="dreamsim_vectors",
            vector_column="dreamsim_vector_blob",
            compute_func=compute,
        )

    # ----------- Main Search Method ----------- #
    def search_similar_images(self, query_image_path: str, index_type: str = "color"):
        """
        Runs a similarity search for a query image and displays the top results.

        Args:
            query_image_path (str): Absolute path to the query image.
            index_type (str): One or more comma-separated index types (e.g. "color,sift").

        Side Effects:
            - Displays a matplotlib plot with top-K matching images and distances.
        """
        path_rel = str(Path(query_image_path).resolve().relative_to(self.images_root))
        ordered = self._get_ordered_index_types(index_type)
        if not ordered:
            return
        query_vec = self._extract_query_vector(path_rel, ordered)
        if query_vec is None:
            return
        canonical = "_".join(ordered)
        index, offset_table = self._load_faiss_index(canonical)
        if index is None:
            return
        distances, indices = index.search(query_vec, self.top_k)
        results = self._fetch_results(indices, distances, offset_table)
        if not results:
            logging.error("Keine ähnlichen Bilder gefunden.")
            return
        self._plot_results(query_image_path, results)

    def _get_ordered_index_types(self, index_type: str):
        """
        Validates and orders the requested index types.

        Args:
            index_type (str): Comma-separated string of feature types.

        Returns:
            List[str]: Valid feature types in canonical order.
        """
        requested = [x.strip() for x in index_type.lower().split(",")]
        valid = ["color", "hog", "lpips", "dreamsim", "sift", "color_sift", "sift_dreamsim"]
        ordered = [v for v in valid if v in requested]
        if not ordered:
            logging.error(f"Unbekannter index_type '{index_type}'. Wähle aus {valid}.")
            return []
        logging.info(f"🔍 Extrahiere Features in Reihenfolge: {ordered}")
        return ordered

    def _extract_query_vector(self, path_rel, ordered):
        """
        Combines individual feature vectors into a query vector.

        Args:
            path_rel (str): Relative path to the query image.
            ordered (List[str]): List of feature types to extract.

        Returns:
            np.ndarray or None: Concatenated query vector.
        """
        parts = []
        for vec_type in ordered:
            if vec_type == "color":
                parts.append(self.extract_color_features(path_rel))
            elif vec_type == "sift":
                parts.append(self.extract_sift_vlad_features(path_rel))
            elif vec_type == "dreamsim":
                parts.append(self.extract_dreamsim_features(path_rel))
            else:
                logging.error(f"Unbekannter Vektor-Typ '{vec_type}' für '{path_rel}'.")
                return None
        if len(parts) == 1:
            query_vec = parts[0].astype("float32")
        else:
            parts = [x.reshape(1, -1) if x.ndim == 1 else x for x in parts]
            query_vec = np.concatenate(parts, axis=1).astype("float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        return query_vec

    def _load_faiss_index(self, canonical):
        """
        Loads a FAISS index and its corresponding offset table.

        Args:
            canonical (str): Canonical name representing the feature combination.

        Returns:
            Tuple[faiss.Index, str]: Loaded index object and offset table name.
        """
        index_file = f"index_hnsw_{canonical}.faiss"
        offset_table = f"faiss_index_offsets_{canonical}"
        try:
            index = faiss.read_index(index_file)
            logging.info(f"Geladener FAISS-Index '{index_file}' mit {index.ntotal} Vektoren.")
            return index, offset_table
        except Exception as e:
            logging.error(f"Fehler beim Laden des Index '{index_file}': {e}")
            return None, None

    def _fetch_results(self, indices, distances, offset_table):
        """
        Maps FAISS index search results back to image paths using offset mapping.

        Args:
            indices (np.ndarray): Indices returned from FAISS search.
            distances (np.ndarray): Corresponding distances from FAISS search.
            offset_table (str): Table storing image_id-to-index offset mapping.

        Returns:
            List[Tuple[Path, float]]: List of (image path, distance) pairs.
        """
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
            full_path = Path(self.base_dir) / "images_v3", fp_row[0]
            results.append((full_path, float(distances[0, rank])))
        conn.close()
        results.sort(key=lambda x: x[1])
        return results

    def _plot_results(self, query_image_path, results):
        """
        Displays a query image alongside the retrieved similar images.

        Args:
            query_image_path (str): Path to the query image.
            results (List[Tuple[Path, float]]): List of paths and distances to similar images.

        Side Effects:
            - Shows a matplotlib figure with the query and result images.
        """
        sns.set_theme(style="whitegrid")
        max_per_row = 3
        total_images = len(results) + 1
        ncols = max_per_row
        nrows = int(np.ceil(total_images / ncols))
        _ , axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
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
    """
    Example usage: Load the recommender and search for images similar to a given file.
    """
    query_image = (
        r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3\image_data\weather_image_recognition\rainbow\0594.jpg"
    )
    rec = ImageRecommender()
    rec.search_similar_images(query_image, index_type="color")