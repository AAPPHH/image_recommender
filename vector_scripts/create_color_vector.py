import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import cv2

try:
    from creat_vector_base import BaseVectorIndexer, load_image
except ImportError:
    from vector_scripts.creat_vector_base import BaseVectorIndexer, load_image

class ColorVectorIndexer(BaseVectorIndexer):
    table_name = "color_vectors"
    vector_column = "color_vector_blob"
    id_column = "image_id"
    bins = 16

    @staticmethod
    def compute_color_vector_worker(args):
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
            return None
        chans = cv2.split(img)
        hists = [cv2.calcHist([c], [0], None, [bins], [0, 256]).flatten() for c in chans]
        vec = np.concatenate(hists).astype(np.float32)
        l2 = np.linalg.norm(vec)
        if l2 != 0:
            vec /= l2
        return vec

    def compute_vectors(self, paths: list[str], chunksize=16):
        images_dir = self.base_dir / "images_v3"
        bins = self.bins
        load_image_kwargs = dict(
            img_size=None,
            gray=False,
            normalize=False,
            use_cv2=True,
        )
        args = [(path, images_dir, bins, load_image_kwargs) for path in paths]
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as exe:
            for vec in exe.map(self.compute_color_vector_worker, args, chunksize=chunksize):
                results.append(vec)
        return results

if __name__ == "__main__":
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = ColorVectorIndexer(
        db_path,
        base_dir,
        log_file="color_indexer.log",
        batch_size=16384
    )
    indexer.run()