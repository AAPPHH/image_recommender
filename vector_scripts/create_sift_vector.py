import os
os.environ["OMP_NUM_THREADS"] = "1"      # OpenMP
os.environ["OPENBLAS_NUM_THREADS"] = "1" # OpenBLAS
os.environ["MKL_NUM_THREADS"] = "1"      # Intel MKL
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # macOS/Accelerate
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import random
from pathlib import Path
import numpy as np
import cv2
import faiss
from multiprocessing import shared_memory, get_context
from concurrent.futures import ProcessPoolExecutor

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
    descriptor_dim = 128
    pca_path = "sift_vlad_pca.joblib"
    pca = None
    pca_dim = 512

    _worker_codebook = None
    _worker_index = None
    _worker_sift = None
    _worker_shm = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = self.base_dir / "images_v3"
        self.codebook = self.load_or_create_codebook(self.images_dir)
        if os.path.exists(self.pca_path):
            self.pca = self.load_pca()
        else:
            self._log_and_print("Kein PCA gefunden, starte automatisches PCA-Training...", level="info")
            self.train_pca_on_sample(pca_sample_size=50_000, batch_size=512)

    @classmethod
    def random_sample_paths_batch_generator(cls, images_dir, sample_size=50_000, batch_size=512, img_formats=(".jpg", ".jpeg", ".png")):
        reservoir = []
        images_dir = Path(images_dir)
        idx = 0
        for ext in img_formats:
            for img_path in images_dir.rglob(f"*{ext}"):
                rel_path = img_path.relative_to(images_dir)
                if idx < sample_size:
                    reservoir.append(rel_path)
                else:
                    r = random.randint(0, idx)
                    if r < sample_size:
                        reservoir[r] = rel_path
                idx += 1
        reservoir = list(dict.fromkeys(reservoir))
        for start in range(0, len(reservoir), batch_size):
            yield reservoir[start:start + batch_size]

    @classmethod
    def load_or_create_codebook(cls, images_dir="images_v3", n_clusters=100_000, sample_limit=10_000_000, sample_size=1_000_000, batch_size=512):
        if cls.codebook is not None:
            return cls.codebook

        if os.path.exists(cls.codebook_path):
            cls.codebook = np.load(cls.codebook_path)
            cls.n_clusters = cls.codebook.shape[0]
            print(f"✅ Codebook loaded from {cls.codebook_path}")
            return cls.codebook

        print("⚠️ Codebook not found! Creating new codebook...")

        img_formats = (".jpg", ".jpeg", ".png")
        sift_img_size = cls.sift_img_size

        total_descs = 0
        n_processed = 0
        all_descs = []

        batch_gen = cls.random_sample_paths_batch_generator(
            images_dir,
            sample_size=sample_size,
            batch_size=batch_size,
            img_formats=img_formats
        )

        for batch in batch_gen:
            args = [(Path(images_dir) / rel_path, sift_img_size) for rel_path in batch]
            with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
                results = list(exe.map(cls._sift_descriptors_worker, args, chunksize=16))
            for descs in results:
                if descs is not None and len(descs) > 0:
                    all_descs.append(descs)
                    total_descs += len(descs)
            n_processed += len(batch)
            print(f"Processed {n_processed} images, total {total_descs} descriptors.")
            if total_descs >= sample_limit:
                print(f"Sample limit {sample_limit} reached. Stopping collection.")
                break

        if not all_descs:
            raise RuntimeError("Keine Deskriptoren gefunden!")

        train_descs = np.vstack(all_descs)[:sample_limit]
        print("Starte FAISS KMeans mit", train_descs.shape[0], "Deskriptoren...")
        os.environ["OMP_NUM_THREADS"] = "24"  
        kmeans = faiss.Kmeans(train_descs.shape[1], n_clusters, niter=25, verbose=True)
        kmeans.train(train_descs)
        centroids = kmeans.centroids
        np.save(cls.codebook_path, centroids)
        cls.codebook = centroids
        cls.n_clusters = centroids.shape[0]
        os.environ["OMP_NUM_THREADS"] = "1" 
        print(f"✅ Codebook created and saved as {cls.codebook_path}")
        return cls.codebook




    @staticmethod
    def _sift_descriptors_worker(args):
        img_path, sift_img_size = args
        img = load_image(img_path, img_size=sift_img_size, use_cv2=True, gray=True, normalize=True, antialias=True)
        if img is None:
            return None
        img8 = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
        sift = cv2.SIFT_create(nfeatures=1000)
        _, descs = sift.detectAndCompute(img8, None)
        if descs is not None and len(descs) > 0:
            descs = descs / (np.linalg.norm(descs, ord=1, axis=1, keepdims=True) + 1e-7)
            descs = np.sqrt(descs)
            descs = descs / (np.linalg.norm(descs, axis=1, keepdims=True) + 1e-7)
            return descs
        return None

    def load_pca(self):
        if self.pca is None:
            import faiss
            self.pca = faiss.read_VectorTransform(self.pca_path)
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
        index.hnsw.efSearch = 128
        index.hnsw.efConstruction = 200
        index.add(cls._worker_codebook.astype(np.float32))

        cls._worker_index = index
        cls._worker_sift = cv2.SIFT_create(nfeatures=4000)


    @classmethod
    def _worker_finalize(cls):
        """Shared-Memory in jedem Worker sauber schließen."""
        if cls._worker_shm is not None:
            cls._worker_shm.close()
            cls._worker_shm = None


    @staticmethod
    def _clean_up_shm(shm):
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


    def train_pca_on_sample(self, pca_sample_size=50_000, batch_size=512):
        images_dir = "images_v3"
        batch_gen = self.random_sample_paths_batch_generator(images_dir, pca_sample_size, batch_size)
        self._log_and_print(f"Starte PCA-Training auf zufälligem Sample von {pca_sample_size} Bildern in Batches.", level="info")

        all_vlad_vecs = []
        for batch_idx, batch_paths in enumerate(batch_gen):
            vlad_vecs = self.compute_vectors(batch_paths)
            vlad_vecs = [vec for vec in vlad_vecs if vec is not None]
            if not vlad_vecs:
                self._log_and_print(f"No valid VLAD vectors in batch {batch_idx+1}!", level="error")
                continue
            all_vlad_vecs.extend(vlad_vecs)
            self._log_and_print(f"PCA-Batch {batch_idx + 1} gesammelt: {len(vlad_vecs)} VLADs (total: {len(all_vlad_vecs)})", level="debug")
            if len(all_vlad_vecs) >= pca_sample_size:
                break

        all_vlad_vecs = all_vlad_vecs[:pca_sample_size]
        vlad_matrix = np.stack(all_vlad_vecs).astype(np.float32)

        self._log_and_print(f"Starte FAISS PCA-Training auf Matrix {vlad_matrix.shape}", level="info")
        pca = faiss.PCAMatrix(vlad_matrix.shape[1], self.pca_dim, eigen_power=-0.5, random_rotation=False)
        pca.train(vlad_matrix)
        faiss.write_VectorTransform(pca, self.pca_path)
        self.pca = pca
        self._log_and_print(f"PCA-Training (FAISS) abgeschlossen und gespeichert unter {self.pca_path}", level="info")
        return pca

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
        
        compressed = self.pca.apply_py(np.stack(results))
        compressed /= np.linalg.norm(compressed, axis=1, keepdims=True) + 1e-7
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