from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import logging
import cv2
import numpy as np

from creat_vector_base import BaseVectorIndexer


class HOGVectorIndexer(BaseVectorIndexer):
    vector_column = "hog_vector_blob"
    hog_img_size = (64, 128)

    def get_pending_rows(self, last_id: int):
        cursor = self.read_conn.cursor()
        sql = (
            "SELECT id, path FROM images "
            "WHERE hog_vector_blob IS NULL AND id > ? "
            "ORDER BY id ASC LIMIT ?"
        )
        return cursor.execute(sql, (last_id, self.batch_size)).fetchall()

    @staticmethod
    def _compute_hog(args):
        rec_idx, rel_path, images_dir, img_size = args
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = images_dir / img_path
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ Image not found: {img_path}")
            logging.warning(f"Image not found: {img_path}")
            return None
        img = cv2.resize(img, img_size)
        hog = cv2.HOGDescriptor()
        vec = hog.compute(img)
        if vec is not None:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.flatten().astype(np.float16)
        else:
            return None


    def compute_vectors(self, paths: list[str]):
        images_dir = self.base_dir / "images_v3"
        args = [
            (idx, rel, images_dir, self.hog_img_size)
            for idx, rel in enumerate(paths)
        ]
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            for vec in executor.map(self._compute_hog, args):
                results.append(vec)
        return results

if __name__ == "__main__":
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = HOGVectorIndexer(
        db_path,
        base_dir,
        log_file="hog_indexer.log",
        batch_size=16384,
    )
    indexer.run()
