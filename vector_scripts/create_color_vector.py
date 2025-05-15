from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import logging
import cv2
import numpy as np

from vector_scripts.creat_vector_base import BaseVectorIndexer


class ColorVectorIndexer(BaseVectorIndexer):
    """
    Erstellt Farb-Histogramm-Vektoren per Channel.
    """
    vector_column = "color_vector_blob"
    bins = 16

    def get_pending_rows(self, last_id: int):
        cursor = self.read_conn.cursor()
        sql = (
            "SELECT id, path FROM images "
            "WHERE color_vector_blob IS NULL AND id > ? "
            "ORDER BY id ASC LIMIT ?"
        )
        return cursor.execute(sql, (last_id, self.batch_size)).fetchall()

    @staticmethod
    def _compute_color(args):
        rec_idx, rel_path, images_dir, bins = args
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = images_dir / img_path
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️ Image not found: {img_path}")
            logging.warning(f"Image not found: {img_path}")
            return None
        feats = []
        for ch in range(3):
            hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
            cv2.normalize(hist, hist)
            feats.append(hist.flatten())
        vec = np.concatenate(feats)
        return vec

    def compute_vectors(self, paths: list[str]):
        images_dir = self.base_dir / "images_v3"
        args = [
            (idx, rel, images_dir, self.bins)
            for idx, rel in enumerate(paths)
        ]
        results = []
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
            for vec in executor.map(self._compute_color, args):
                results.append(vec)
        return results


if __name__ == "__main__":
    db_path = "images.db"
    base_dir = Path().cwd()
    indexer = ColorVectorIndexer(
        db_path,
        base_dir,
        log_file="color_indexer.log"
    )
    indexer.run()