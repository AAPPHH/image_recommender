import os
import random
from pathlib import Path
import numpy as np
import cv2
import faiss
from multiprocessing import shared_memory, get_context
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import joblib

# Dummy-Import für load_image etc.
try:
    from creat_vector_base import BaseVectorIndexer, load_image
except ImportError:
    from vector_scripts.creat_vector_base import BaseVectorIndexer, load_image

class SIFTVLADVectorIndexer(BaseVectorIndexer):
    table_name = "sift_vectors"
    vector_column = "sift_vector_blob"
    id_column = "image_id"
    sift_img_size = (512, 512)
    codebook_path = "sift_codebook.npy"
    codebook = None
    n_clusters = 512
    descriptor_dim = 128
    pca_path = "sift_vlad_pca.joblib"
    pca = None
    pca_dim = 256

    _worker_codebook = None
    _worker_index = None
    _worker_sift = None
    _worker_shm = None

    os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
    os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
    os.environ["MKL_NUM_THREADS"] = "1"      # Intel MKL
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # macOS/Accelerate
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = self.base_dir / "images_v3"
        self.codebook = self.load_or_create_codebook(self.images_dir, self.n_clusters)
        if os.path.exists(self.pca_path):
            self.pca = self.load_pca()
        else:
            self._log_and_print("Kein PCA gefunden, starte automatisches PCA-Training...", level="info")
            self.train_pca_on_sample(pca_sample_size=50_000, batch_size=512)

    @classmethod
    def load_or_create_codebook(cls, images_dir="images_v3", n_clusters=256, sample_limit=500000):
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
            sift = cv2.SIFT_create(nfeatures=4000)
            for img_path in img_paths:
                img = load_image(img_path, img_size=cls.sift_img_size, use_cv2=True, gray=True, normalize=True, antialias=True)
                if img is None:
                    continue
                img8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
                _, descs = sift.detectAndCompute(img8, None)
                if descs is not None and len(descs) > 0:
                    descs = descs / (np.linalg.norm(descs, ord=1, axis=1, keepdims=True) + 1e-7)
                    descs = np.sqrt(descs)
                    descs = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-7)
                    all_descs.append(descs)
                if sum(len(x) for x in all_descs) > sample_limit:
                    break

            if not all_descs:
                raise RuntimeError("Keine Deskriptoren gefunden!")

            all_descs = np.vstack(all_descs)[:sample_limit]
            print("Starte KMeans mit", all_descs.shape, "...")
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=16384, verbose=1, init="k-means++" ).fit(all_descs)
            np.save(cls.codebook_path, kmeans.cluster_centers_)
            cls.codebook = kmeans.cluster_centers_
            cls.n_clusters = cls.codebook.shape[0]
            print(f"✅ Codebook created and saved as {cls.codebook_path}")
            return cls.codebook


    def load_pca(self):
        if self.pca is None:
            self.pca = joblib.load(self.pca_path)
            self._log_and_print(f"Loaded PCA from {self.pca_path}", level="info")
        return self.pca


    @classmethod
    def _worker_init(cls, shm_name, codebook_shape, codebook_dtype):
        """Wird einmal pro Worker aufgerufen."""
        # Shared-Memory öffnen und als Array mappen (Zero-Copy!)
        cls._worker_shm = shared_memory.SharedMemory(name=shm_name)
        cls._worker_codebook = np.ndarray(codebook_shape, dtype=codebook_dtype, buffer=cls._worker_shm.buf)

        d = cls._worker_codebook.shape[1]
        hnsw_m = 48
        index = faiss.IndexHNSWFlat(d, hnsw_m)
        index.hnsw.efSearch = 64
        index.hnsw.efConstruction = 200
        index.add(cls._worker_codebook.astype(np.float32))

        cls._worker_index = index
        cls._worker_sift = cv2.SIFT_create(nfeatures=1000)


    @classmethod
    def _worker_finalize(cls):
        """Shared-Memory in jedem Worker sauber schließen."""
        if cls._worker_shm is not None:
            cls._worker_shm.close()
            cls._worker_shm = None

    @classmethod
    def _compute_sift_vlad_worker(cls, args):
        idx, rel_path, images_dir, img_size = args

        img_path = images_dir / rel_path
        img = load_image(img_path, img_size=img_size,
                         use_cv2=True, gray=True, normalize=True, antialias=True)
        if img is None:
            return (idx, None)

        img8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
        _, descriptors = cls._worker_sift.detectAndCompute(img8, None)
        n_clusters, descriptor_dim = cls._worker_codebook.shape

        if descriptors is None or len(descriptors) == 0:
            return (idx, np.zeros(n_clusters * descriptor_dim, dtype=np.float32))
        
        descriptors = descriptors / (np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True) + 1e-7)
        descriptors = np.sqrt(descriptors)
        descriptors = descriptors / (np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-7)
        D, I = cls._worker_index.search(descriptors.astype(np.float32), 4)
        sigma = 125.0
        weights = np.exp(-D / (2 * sigma * sigma))
        vlad = np.zeros((n_clusters, descriptor_dim), dtype=np.float32)

        for d_idx, (idxs, ws) in enumerate(zip(I, weights)):
            for i, w in zip(idxs, ws):
                vlad[i] += w * (descriptors[d_idx] - cls._worker_codebook[i])

        row_norms = np.linalg.norm(vlad, axis=1, keepdims=True)
        row_norms[row_norms == 0] = 1
        vlad = vlad / row_norms
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        vlad = vlad.flatten()
        norm = np.linalg.norm(vlad)

        if norm:
            vlad /= norm

        return (idx, vlad)

    @staticmethod
    def _clean_up_shm(shm):
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass

    def random_sample_paths_batch_generator(self, images_dir, sample_size=50_000, batch_size=512, pattern="*.jpg"):
        reservoir = []
        images_dir = Path(images_dir)
        for idx, img_path in enumerate(images_dir.rglob(pattern)):
            rel_path = img_path.relative_to(images_dir)
            if idx < sample_size:
                reservoir.append(rel_path)
            else:
                r = random.randint(0, idx)
                if r < sample_size:
                    reservoir[r] = rel_path
        for start in range(0, len(reservoir), batch_size):
            yield reservoir[start:start + batch_size]


    def train_pca_on_sample(self, pca_sample_size=50_000, batch_size=512):
        images_dir = "images_v3"
        batch_gen = self.random_sample_paths_batch_generator(images_dir, pca_sample_size, batch_size)
        self._log_and_print(f"Starte PCA-Training auf zufälligem Sample von {pca_sample_size} Bildern in Batches.", level="info")

        ipca = IncrementalPCA(n_components=self.pca_dim, whiten=True, batch_size=batch_size)

        for batch_idx, batch_paths in enumerate(batch_gen):
            vlad_vecs = self.compute_vectors(batch_paths)
            # Filter out None results (if compute_vectors returns a mix)
            vlad_vecs = [vec for vec in vlad_vecs if vec is not None]
            if not vlad_vecs:
                self._log_and_print(f"No valid VLAD vectors in batch {batch_idx+1}!", level="error")
                continue
            vlad_matrix = np.stack(vlad_vecs)
            ipca.partial_fit(vlad_matrix)
            self._log_and_print(f"PCA-Batch {batch_idx + 1} fit: {vlad_matrix.shape}", level="debug")


        joblib.dump(ipca, self.pca_path)
        self.pca = ipca
        self._log_and_print(f"PCA-Training (Incremental) abgeschlossen und gespeichert unter {self.pca_path}", level="info")
        return ipca

    # ----------------------------------------------
    # Hauptmethode
    # -----------------------------------------------

    def compute_vectors(self, paths: list[str]):
        codebook_f32 = np.ascontiguousarray(self.codebook.astype(np.float32))
        shm = shared_memory.SharedMemory(create=True, size=codebook_f32.nbytes)
        shm_buf = np.ndarray(codebook_f32.shape, dtype=codebook_f32.dtype, buffer=shm.buf)
        shm_buf[:] = codebook_f32

        args = [(i, rel, self.images_dir, self.sift_img_size) for i, rel in enumerate(paths)]
        result_dict = {}
        self._log_and_print(f"Processing {len(args)} images with SIFT-VLAD", level="info")

        ctx = get_context("spawn") if os.name == "nt" else get_context("fork")

        with ProcessPoolExecutor(
            max_workers=os.cpu_count() or 1,
            mp_context=ctx,
            initializer=self.__class__._worker_init,
            initargs=(shm.name, codebook_f32.shape, codebook_f32.dtype)
        ) as exe:
            for idx, vec in exe.map(self.__class__._compute_sift_vlad_worker, args, chunksize=16):
                if vec is not None:
                    result_dict[idx] = vec

        self._clean_up_shm(shm)

        valid_indices = sorted(result_dict)
        results = [result_dict[i] for i in valid_indices]
        if self.pca is None:
            self._log_and_print("PCA not initialized, returning raw VLAD vectors.", level="warning")
            return [vec.astype(np.float32) for vec in results]
        
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
        batch_size=512,
    )
    indexer.run() 