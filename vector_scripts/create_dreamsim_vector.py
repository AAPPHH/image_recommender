import numpy as np
from pathlib import Path
import torch
from dreamsim import dreamsim

try:
    from vector_scripts.create_vector_base import BaseVectorIndexer, load_image
except ImportError:
    from vector_scripts.create_vector_base import BaseVectorIndexer, load_image

class DreamSimVectorIndexer(BaseVectorIndexer):
    table_name = "dreamsim_vectors"             # NEU: Deine Tabelle!
    vector_column = "dreamsim_vector_blob"      # Name des BLOB-Feldes
    id_column = "image_id" 

    def __init__(
        self,
        db_path: str,
        base_dir: str,
        batch_size: int = 4096,
        model_batch: int = 128,
        log_file: str = "dreamsim_indexer.log",
        log_dir: str = "logs",
    ):
        super().__init__(db_path, base_dir, batch_size, log_file, log_dir)
        self.model_batch = model_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_and_print(f"Using device: {self.device}", level="info")
        self._setup_model()

    def _setup_model(self):
        """
        Loads and prepares the DreamSim model on the selected device.

        Initializes the model, sets it to eval mode, and performs a dummy
        forward pass to determine output dimensionality.
        """
        self.model, self.preprocess = dreamsim(
            pretrained=True,
            device=self.device,
            normalize_embeds=True,
            dreamsim_type="ensemble",
        )
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            emb = self.model.embed(dummy)
        self.dim = emb.shape[-1]
        self._log_and_print("DreamSim model loaded and warmed up.", level="info")

    def _batch_image_to_vector(self, image_paths: list[str]) -> tuple[torch.Tensor, list[str]]:
        """
        Converts a list of image paths into a batch of DreamSim embeddings.

        Args:
            image_paths (list[str]): List of relative image paths.

        Returns:
            tuple:
                - torch.Tensor: Embedding tensor of shape (N, D).
                - list[str]: List of image paths that were successfully processed.
        """
        images = []
        valid_paths = []
        for rel_path in image_paths:
            full_path = self.base_dir / "images_v3" / rel_path
            img = load_image(full_path, img_size=(224, 224), gray=False, normalize=True, antialias=True)
            if img is None:
                self._log_and_print(f"⚠️ Error loading {full_path}", level="warning")
                continue
            try:
                tensor = self.preprocess(img)
                if tensor.ndim == 4 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if not (isinstance(tensor, torch.Tensor) and tensor.shape == (3, 224, 224)):
                    self._log_and_print(
                        f"❗️ Preprocess returned wrong shape for {full_path}: {tensor.shape}",
                        level="warning"
                    )
                    continue
                images.append(tensor)
                valid_paths.append(rel_path)
            except Exception as e:
                self._log_and_print(f"⚠️ Preprocessing failed for {full_path}: {e}", level="warning")
                continue
        if not images:
            return torch.empty(0, self.dim), []
        image_tensor = torch.stack(images).to(self.device)
        print("Batch tensor shape:", image_tensor.shape)
        with torch.no_grad():
            embeddings = self.model.embed(image_tensor)
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings.cpu(), valid_paths


    def compute_vectors(self, paths: list[str]) -> list[np.ndarray | None]:
        """
        Computes a DreamSim embedding for each image path.

        Args:
            paths (List[str]): List of relative image paths.

        Returns:
            List[np.ndarray | None]: List of float32 arrays (or None if processing failed).
        """
        results: list[np.ndarray | None] = [None] * len(paths)
        total_batches = (len(paths) + self.model_batch - 1) // self.model_batch
        for batch_idx, start in enumerate(range(0, len(paths), self.model_batch)):
            chunk = paths[start : start + self.model_batch]
            embeddings, valid_paths = self._batch_image_to_vector(chunk)
            for j, rel in enumerate(chunk):
                if rel in valid_paths:
                    idx = valid_paths.index(rel)
                    results[start + j] = embeddings[idx].numpy().astype('float32')
                else:
                    results[start + j] = None
            self._log_and_print(
                f"✅ Sub-batch {batch_idx+1}/{total_batches} processed: {len(chunk)} images",
                level="info"
            )
        self._log_and_print(
            f"👍 All {total_batches} sub-batches processed.",
            level="info"
        )
        return results

if __name__ == "__main__":
    """
    CLI entry point to run DreamSim vector indexing on all images in the database.
    """
    db_path = "images.db"
    base_dir = Path().cwd()

    dreamsim_indexer = DreamSimVectorIndexer(
        db_path,
        base_dir,
        batch_size=4096,
        model_batch=128,
        log_file="dreamsim_indexer.log",
        log_dir="logs",
    )
    dreamsim_indexer.run()