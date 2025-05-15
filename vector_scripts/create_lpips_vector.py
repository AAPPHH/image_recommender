import numpy as np
import torch
import lpips
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from torch.nn.functional import adaptive_avg_pool2d, normalize
import gc

from vector_scripts.creat_vector_base import BaseVectorIndexer

class LPIPSVectorIndexer(BaseVectorIndexer):
    vector_column = "lpips_vector_blob"

    def __init__(
        self,
        db_path: str,
        base_dir: str,
        batch_size: int = 8192,
        model_batch: int = 128,
        log_file: str = "lpips_indexer.log",
        log_dir: str = "logs",
    ):
        super().__init__(db_path, base_dir, batch_size, log_file, log_dir)
        self.model_batch = model_batch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_and_print(f"Using device: {self.device}", level="info")
        self.net = lpips.LPIPS(net="vgg").to(self.device).half().eval()
        self._log_and_print("LPIPS model loaded.", level="info")

    def get_pending_rows(self, last_id: int):
        cursor = self.read_conn.cursor()
        sql = (
            "SELECT id, path FROM images "
            "WHERE lpips_vector_blob IS NULL AND id > ? "
            "ORDER BY id ASC LIMIT ?"
        )
        return cursor.execute(sql, (last_id, self.batch_size)).fetchall()

    def _prepare_tensors(self, paths: list[str]) -> list[torch.Tensor | None]:
        tensors: list[torch.Tensor | None] = []
        for rel_path in paths:
            img_path = self.base_dir / "images_v3" / rel_path
            if not img_path.exists():
                self._log_and_print(f"⚠️ Not found: {img_path}", level="warning")
                tensors.append(None)
                continue
            try:
                img = read_image(str(img_path)).float() / 255.0
            except Exception as e:
                self._log_and_print(f"⚠️ Error reading {img_path}: {e}", level="warning")
                tensors.append(None)
                continue
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] == 4:
                img = img[:3]
            elif img.shape[0] != 3:
                self._log_and_print(
                    f"⚠️ Unexpected channels ({img.shape[0]}), skipped.", level="warning"
                )
                tensors.append(None)
                continue
            tensors.append(resize(img, [224, 224], antialias=True))
        return tensors

    def _compute_batches(self, tensors: list[torch.Tensor | None]) -> list[np.ndarray | None]:
        vecs: list[np.ndarray | None] = [None] * len(tensors)
        for i in range(0, len(tensors), self.model_batch):
            batch_idxs = [j for j in range(i, min(i + self.model_batch, len(tensors))) if tensors[j] is not None]
            if not batch_idxs:
                continue
            batch = torch.stack([tensors[j] for j in batch_idxs]).to(self.device).half()
            batch = (batch - 0.5) / 0.5
            with torch.inference_mode():
                feats = self.net.net(batch)
            fmap = feats[-1] if isinstance(feats, (list, tuple)) else feats
            out = normalize(adaptive_avg_pool2d(fmap, 1).flatten(1), dim=1)
            arr = out.cpu().numpy().astype('float32')
            for idx, vec in zip(batch_idxs, arr):
                vecs[idx] = vec
            del batch, feats, fmap, out
            torch.cuda.empty_cache(); gc.collect()
        return vecs

    def compute_vectors(self, paths: list[str]) -> list[np.ndarray | None]:
        tensors = self._prepare_tensors(paths)
        return self._compute_batches(tensors)
    
if __name__ == "__main__":
    import os
    db_path = "images.db"
    base_dir = os.getcwd()
    lpips_indexer = LPIPSVectorIndexer(
        db_path,
        base_dir,
        batch_size=8192,
        model_batch=128,
        log_file="lpips_indexer.log"
    )
    lpips_indexer.run()