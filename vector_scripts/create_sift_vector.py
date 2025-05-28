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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

try:
    from creat_vector_base import BaseVectorIndexer, load_image
except ImportError:
    from vector_scripts.creat_vector_base import BaseVectorIndexer, load_image

class SIFTVLADVectorIndexer(BaseVectorIndexer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table_name = "sift_vectors"
    vector_column = "sift_vector_blob"
    id_column = "image_id"
    sift_img_size = (512, 512)
    codebook_path = "sift_codebook.npy"
    codebook = None
    descriptor_dim = 128
    pca_path = "sift_vlad_pca.joblib"
    pca = None
    pca_dim = 256
    n_clusters = 256

    _worker_codebook = None
    _worker_index = None
    _worker_sift = None
    _worker_shm = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.images_dir = self.base_dir / "images_v3"
        self.codebook = self.load_or_create_codebook(self.images_dir, n_clusters=self.n_clusters, sample_limit=self.n_clusters * 20_000)
        self.index_path = "hnsw.idx"
        self.faiss_index = self.build_or_load_index(self.index_path)

        if getattr(self, "encoder_model", None) is None:
            try:
                self._log_and_print("Kein Encoder gefunden, starte automatisches Autoencoder-Training...", level="info")
                self.train_autoencoder_on_sample(epochs=50, batch_size=self.batch_size)
            except ImportError:
                self._log_and_print("TensorFlow/Keras nicht installiert, kann keinen Autoencoder trainieren!", level="warning")
                self.encoder_model = None

    @classmethod
    def random_sample_paths_batch_generator(
        cls, images_dir, sample_size=50_000, batch_size=512, img_formats=(".jpg", ".jpeg", ".png")
    ):
        images_dir = Path(images_dir)
        all_paths = []
        seen = set()
        for img_path in images_dir.rglob("*"):
            if img_path.suffix.lower() in img_formats:
                rel_path = img_path.relative_to(images_dir)
                if rel_path not in seen:
                    seen.add(rel_path)
                    all_paths.append(rel_path)
        if len(all_paths) > sample_size:
            sample = random.sample(all_paths, sample_size)
        else:
            sample = all_paths
        for start in range(0, len(sample), batch_size):
            yield sample[start : start + batch_size]


    @classmethod
    def load_or_create_codebook(cls, images_dir="images_v3", n_clusters=256, sample_limit=5_120_000, sample_size=1_000_000, batch_size=512):
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

        batch_gen_for_codebook = cls.random_sample_paths_batch_generator(
            images_dir,
            sample_size=sample_size,
            batch_size=batch_size,
            img_formats=img_formats
        )

        for batch in batch_gen_for_codebook:
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
        kmeans = faiss.Kmeans(
            train_descs.shape[1], n_clusters,
            niter=25, verbose=True,
            **({"gpu": True} if hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0 else {})
        )
        kmeans.train(train_descs)
        centroids = kmeans.centroids
        np.save(cls.codebook_path, centroids)
        cls.codebook = centroids
        cls.n_clusters = centroids.shape[0]
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
            self.pca = faiss.read_VectorTransform(self.pca_path)
            self._log_and_print(f"Loaded PCA from {self.pca_path}", level="info")
        return self.pca
    
    
    def build_or_load_index(
        self, 
        index_path="hnsw.idx",
        hnsw_m=48, 
        efConstruction=200, 
        efSearch=128
    ):
        dim = self.codebook.shape[1]
        if not os.path.exists(index_path):
            print("Baue HNSW-Index ...")
            index = faiss.IndexHNSWFlat(dim, hnsw_m)
            index.hnsw.efConstruction = efConstruction
            index.hnsw.efSearch = efSearch
            index.add(self.codebook.astype(np.float32))
            faiss.write_index(index, index_path)
            print(f"✅ HNSW-Index gespeichert als {index_path}")
        else:
            print(f"HNSW-Index existiert schon ({index_path}), lade Index ...")
            index = faiss.read_index(index_path)
            print("✅ Index geladen.")
        return index


    @classmethod
    def _worker_init(cls, shm_name, codebook_shape, codebook_dtype):
        cls._worker_shm = shared_memory.SharedMemory(name=shm_name)
        cls._worker_codebook = np.ndarray(codebook_shape, dtype=codebook_dtype, buffer=cls._worker_shm.buf)

        index_path = "hnsw.idx"
        if not os.path.exists(index_path):
            raise RuntimeError(f"Index-Datei {index_path} nicht gefunden! Bitte vorher bauen.")
        cls._worker_index = faiss.read_index(index_path)
        cls._worker_sift = cv2.SIFT_create(nfeatures=1000)


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


    def train_autoencoder_on_sample(self, sample_size=500_000, batch_size=None, latent_dim=None, epochs=25):
  
        print(f"Using device: {self.device}")

        batch_gen = self.random_sample_paths_batch_generator(
                images_dir=self.images_dir,
                sample_size=sample_size,
                batch_size=batch_size
            )

        first_batch = next(batch_gen)
        vlad_vecs = self.compute_vectors(first_batch)
        vlad_vecs = [vec for vec in vlad_vecs if vec is not None]
        if len(vlad_vecs) == 0:
            raise ValueError("First batch contains no valid VLAD vectors!")
        X_sample = np.stack(vlad_vecs).astype(np.float32)
        input_dim = X_sample.shape[1]
        if latent_dim is None:
            latent_dim = self.pca_dim

        class Autoencoder(nn.Module):
            def __init__(self, input_dim=32768, latent_dim=256, dropout_rate=0.1):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 2048),
                    nn.LayerNorm(2048),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2048, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, latent_dim)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(512, 2048),
                    nn.LayerNorm(2048),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(2048, input_dim)
                )
                self._initialize_weights()


            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def forward(self, x):
                z = self.encoder(x)
                x_recon = self.decoder(z)
                return x_recon, z

        model = Autoencoder(input_dim, latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        self._log_and_print(
            f"Training autoencoder (PReLU, generator, PyTorch) with latent_dim={latent_dim}, epochs={epochs}",
            level="info",
        )

        scaler = GradScaler("cuda")

        for epoch in range(epochs):
            if epoch == 0:
                X = X_sample
            else:
                try:
                    batch_paths = next(batch_gen)
                except StopIteration:
                    self._log_and_print("No data found in epoch", level="warning")
                    continue

                vlad_vecs = self.compute_vectors(batch_paths)
                vlad_vecs = [vec for vec in vlad_vecs if vec is not None]
                if len(vlad_vecs) == 0:
                    continue
                X = np.stack(vlad_vecs).astype(np.float32)
                X_tensor = torch.from_numpy(X).to(self.device)
                optimizer.zero_grad()
                with autocast("cuda"):
                    x_recon, _ = model(X_tensor)
                    loss = criterion(x_recon, X_tensor)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                avg_loss = loss.item()
                self._log_and_print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.5f}", level="info")


        torch.save(model.state_dict(), "sift_vlad_autoencoder.pt")
        torch.save(model.encoder.state_dict(), "sift_vlad_encoder.pt")
        self.encoder_model = model.encoder

        self._log_and_print("Autoencoder-Training abgeschlossen (PReLU, generator, PyTorch) und Encoder gespeichert.", level="info")
        return model.encoder


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

        if getattr(self, "encoder_model", None) is not None:
            self.encoder_model.eval()
            X_tensor = torch.from_numpy(np.stack(results)).float().to(self.device)
            with torch.no_grad():
                compressed = self.encoder_model(X_tensor)
            compressed = compressed.cpu().numpy()
            compressed /= np.linalg.norm(compressed, axis=1, keepdims=True) + 1e-7
            return [vec.astype(np.float32) for vec in compressed]

        self._log_and_print(
            "No Autoencoder initialized, returning raw VLAD vectors.",
            level="warning"
        )
        return [vec.astype(np.float32) for vec in results]

if __name__ == "__main__":
    print("FAISS-GPU verfügbar!" if hasattr(faiss, "StandardGpuResources") else "Nur FAISS-CPU installiert.")
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = SIFTVLADVectorIndexer(
        db_path,
        base_dir,
        log_file="sift_vlad_indexer.log",
        batch_size=4096,
    )
    indexer.run() 