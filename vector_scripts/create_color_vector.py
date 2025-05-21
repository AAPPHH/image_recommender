import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import logging

try:
    from creat_vector_base import BaseVectorIndexer, load_image
except ImportError:
    from vector_scripts.creat_vector_base import BaseVectorIndexer, load_image

def compute_color_worker(args):
    rel_path, images_dir, bins, load_image_kwargs = args
    img_path = Path(rel_path)
    if not img_path.is_absolute():
        img_path = images_dir / img_path
    img = load_image(img_path, **load_image_kwargs)
    if (
        img is None
        or not isinstance(img, np.ndarray)
        or img.ndim != 3
        or img.shape[2] != 3
    ):
        print(f"⚠️ Image unreadable or wrong shape: {img_path}")
        logging.warning(f"Image unreadable or wrong shape: {img_path}")
        return None
    feats = []
    for ch in range(3):
        hist = np.histogram(img[..., ch], bins=bins, range=(0, 1), density=True)[0]
        feats.append(hist)
    vec = np.concatenate(feats).astype(np.float32)
    return vec

class ColorVectorIndexer(BaseVectorIndexer):
    table_name = "color_vectors"
    vector_column = "color_vector_blob"
    id_column = "image_id"
    bins = 64

    def compute_vectors(self, paths: list[str]):
        images_dir = self.base_dir / "images_v3"
        load_image_kwargs = dict(
            img_size=None,
            gray=False,
            normalize=True,
            to_numpy=True,
            antialias=True,
        )
        args = [
            (rel, images_dir, self.bins, load_image_kwargs)
            for rel in paths
        ]
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            for vec in executor.map(compute_color_worker, args):
                results.append(vec)
        return results

if __name__ == "__main__":
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = ColorVectorIndexer(
        db_path,
        base_dir,
        log_file="color_indexer.log",
        batch_size = 16384
    )
    indexer.run()