import signal
import sys
from pathlib import Path
import sqlite3
import pickle
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor
import os

from logging_utils import setup_logging, console_and_log


def compute_color_worker(args):
    rec_id, rel_path, base_folder_str, bins = args
    base_folder = Path(base_folder_str)
    p = Path(rel_path)
    img_path = p if p.is_absolute() else base_folder / p

    img = cv2.imread(str(img_path))
    if img is None:
        console_and_log(f"⚠️ Image not found: {img_path}", level="warning")
        return rec_id, None

    feats = []
    for ch in range(3):
        hist = cv2.calcHist([img], [ch], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        feats.append(hist.flatten())
    vec = np.concatenate(feats)
    blob = pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL)
    return rec_id, blob


class ColorVectorCreatorDB:
    def __init__(
        self,
        db_path: str,
        base_folder: str,
        bins: int = 16,
        batch_size: int = 1000,
        workers: int = None,
    ):
        # Initialize logging
        setup_logging("color_indexer.log")

        self.db_path = db_path
        self.base_folder = Path(base_folder)
        self.bins = bins
        self.batch_size = batch_size
        self.workers = workers or os.cpu_count() or 1

        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        console_and_log("⚠️ Interrupted by user. Exiting…", level="info")
        sys.exit(0)

    def update_db(self):
        read_conn = sqlite3.connect(self.db_path)
        read_cur = read_conn.cursor()
        read_conn.execute("PRAGMA journal_mode=WAL;")
        read_cur.execute(
            "SELECT id, path FROM images WHERE color_vector_blob IS NULL"
        )

        write_conn = sqlite3.connect(self.db_path)
        write_conn.execute("PRAGMA journal_mode=WAL;")
        write_conn.execute("PRAGMA synchronous=OFF;")
        write_conn.execute("PRAGMA temp_store=MEMORY;")
        write_cur = write_conn.cursor()

        total = write_cur.execute(
            "SELECT COUNT(*) FROM images WHERE color_vector_blob IS NULL"
        ).fetchone()[0]
        console_and_log(f"📦 Images to process: {total}", level="info")

        processed = 0
        buffer = []

        with ProcessPoolExecutor(max_workers=self.workers) as exe:
            while True:
                rows = read_cur.fetchmany(self.batch_size)
                if not rows:
                    break

                args = [
                    (rid, rel, str(self.base_folder), self.bins) for rid, rel in rows
                ]

                for rid, blob in exe.map(compute_color_worker, args):
                    if blob is not None:
                        buffer.append((blob, rid))
                    processed += 1

                write_cur.executemany(
                    "UPDATE images SET color_vector_blob = ? WHERE id = ?", buffer
                )
                write_conn.commit()
                console_and_log(
                    f"✅ Batch completed: {processed}/{total}", level="info"
                )
                buffer.clear()

        console_and_log(
            f"🏁 Finished: {processed}/{total} images processed.", level="info"
        )

        read_conn.close()
        write_conn.close()


if __name__ == "__main__":
    creator = ColorVectorCreatorDB(
        db_path="images.db",
        base_folder=r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3",
        bins=16,
        batch_size=10000,
        workers=None,
    )
    creator.update_db()
    console_and_log("✅ Color vectors created and saved.", level="info")