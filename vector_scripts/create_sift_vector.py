import os
import random
from pathlib import Path
import cv2
import joblib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


from creat_vector_base import BaseVectorIndexer, load_image

class SIFTVLADVectorIndexer(BaseVectorIndexer):
    table_name = "sift_vectors"
    vector_column = "sift_vector_blob"
    id_column = "image_id"
    sift_img_size = (256, 256)
    codebook_path = "sift_codebook.npy"
    codebook = None
    n_clusters = 256
    descriptor_dim = 128
    pca_path = "sift_vlad_pca.joblib"
    pca = None
    pca_dim = 256

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = self.base_dir / "images_v3"
        self.codebook = self.load_or_create_codebook(self.images_dir, self.n_clusters)
        if os.path.exists(self.pca_path):
            self.pca = self.load_pca()
        else:
            self.pca = None

    @classmethod
    def load_or_create_codebook(cls, images_dir="images_v3", n_clusters=256, sample_limit=200000):
        if cls.codebook is not None:
            return cls.codebook

        if os.path.exists(cls.codebook_path):
            cls.codebook = np.load(cls.codebook_path)
            cls.n_clusters = cls.codebook.shape[0]
            print(f"✅ Codebook loaded from {cls.codebook_path}")
            return cls.codebook
        else:
            print("⚠️ Codebook not found! Creating new codebook...")

            images_dir = Path(images_dir)
            img_paths = list(images_dir.rglob("*.jpg"))
            random.shuffle(img_paths)

            all_descs = []
            for img_path in img_paths:
                img = load_image(img_path, img_size=cls.sift_img_size, gray=True, to_numpy=True)
                if img is None:
                    continue
                img8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
                sift = cv2.SIFT_create()
                _, descs = sift.detectAndCompute(img8, None)
                if descs is not None and len(descs) > 0:
                    all_descs.append(descs)
                if sum(len(x) for x in all_descs) > sample_limit:
                    break

            if not all_descs:
                raise RuntimeError("Keine Deskriptoren gefunden!")

            all_descs = np.vstack(all_descs)[:sample_limit]

            print("Starte KMeans mit", all_descs.shape, "...")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, verbose=1).fit(all_descs)
            np.save(cls.codebook_path, kmeans.cluster_centers_)
            cls.codebook = kmeans.cluster_centers_
            cls.n_clusters = cls.codebook.shape[0]
            print(f"✅ Codebook created and saved as {cls.codebook_path}")
            return cls.codebook

    def train_and_save_pca(self, vlad_matrix):
        pca = PCA(n_components=self.pca_dim, whiten=True, random_state=42)
        pca.fit(vlad_matrix)
        joblib.dump(pca, self.pca_path)
        self._log_and_print(f"Trained and saved PCA to {self.pca_path}", level="info")
        return pca

    def load_pca(self):
        if self.pca is None:
            self.pca = joblib.load(self.pca_path)
            self._log_and_print(f"Loaded PCA from {self.pca_path}", level="info")
        return self.pca

    @staticmethod
    def _compute_sift_vlad(args):
        idx, rel_path, images_dir, codebook, img_size = args
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = images_dir / img_path

        img = load_image(img_path, img_size=img_size, gray=True, to_numpy=True)
        if img is None:
            return (idx, None)

        img8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
        sift = cv2.SIFT_create()
        _, descriptors = sift.detectAndCompute(img8, None)

        n_clusters, descriptor_dim = codebook.shape
        if descriptors is None or len(descriptors) == 0:
            return (idx, np.zeros(n_clusters * descriptor_dim, dtype=np.float32))

        idxs = np.argmin(cdist(descriptors, codebook), axis=1)
        vlad = np.zeros((n_clusters, descriptor_dim), dtype=np.float32)
        for i, d in zip(idxs, descriptors):
            vlad[i] += d - codebook[i]

        row_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        vlad = vlad / row_norms

        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad = vlad.flatten()
        norm = np.linalg.norm(vlad)
        if norm:
            vlad /= norm 
        return (idx, vlad)

    def compute_vectors(self, paths: list[str]):
        args = [
            (idx, rel, self.images_dir, self.codebook, self.sift_img_size)
            for idx, rel in enumerate(paths)
        ]

        result_dict = {}
        self._log_and_print(f"Processing {len(args)} images with SIFT-VLAD...", level="info")

        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
            for idx, vec in exe.map(self._compute_sift_vlad, args, chunksize=256):
                if vec is not None:
                    result_dict[idx] = vec

        valid_indices = sorted(result_dict)
        results = [result_dict[i] for i in valid_indices]

        if self.pca is None:
            if len(results) < self.pca_dim:
                self._log_and_print(
                    f"Not enough samples for PCA (got {len(results)}, need ≥{self.pca_dim}). "
                    "Waiting for next batch…", level="warning")
                return results
            vlad_matrix = np.stack(results)
            self.pca = self.train_and_save_pca(vlad_matrix)
            compressed = self.pca.transform(vlad_matrix)
        else:
            compressed = self.pca.transform(np.stack(results))
            
        compressed = normalize(compressed, axis=1)

        return [vec.astype(np.float32) for vec in compressed]

if __name__ == "__main__":
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = SIFTVLADVectorIndexer(
        db_path,
        base_dir,
        log_file="sift_vlad_indexer.log",
        batch_size=16384,
    )
    indexer.run()
