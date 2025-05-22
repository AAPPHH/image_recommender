import signal
import sys
import sqlite3
import pickle
import logging
from pathlib import Path
from PIL import Image
import numpy as np


class BaseVectorIndexer:
    vector_column: str = None
    batch_size: int = 1024
    table_name: str = None
    path_column: str = "path"
    id_column: str = "image_id" 

    def __init__(
        self,
        db_path: str,
        base_dir: str,
        batch_size: int = None,
        log_file: str = "vector_indexer.log",
        log_dir: str = "logs",
    ):
        self.db_path = db_path
        self.base_dir = Path(base_dir)
        if batch_size is not None:
            self.batch_size = batch_size

        self._setup_logging(log_file, log_dir)
        self._init_db()

        signal.signal(signal.SIGINT, self._handle_sigint)

    def _setup_logging(self, log_file: str, log_dir: str):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        full_path = Path(log_dir) / log_file

        logging.basicConfig(
            level=logging.INFO,
            filename=str(full_path),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )
        self._log_and_print(f"Logging initialized at {full_path}", level="info")

    def _log_and_print(self, message: str, level: str = "info"):
        print(message)
        lvl = level.lower()
        if lvl == "info":
            logging.info(message)
        elif lvl == "warning":
            logging.warning(message)
        elif lvl == "error":
            logging.error(message)
        else:
            logging.debug(message)

    def _handle_sigint(self, signum, frame):
        self._log_and_print("⚠️ Aborted by user.", level="info")
        sys.exit(0)

    def _init_db(self):
        self.read_conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self.read_conn.execute("PRAGMA journal_mode=WAL;")
        self.write_conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        self.write_conn.execute("PRAGMA journal_mode=WAL;")
        self.write_conn.execute("PRAGMA synchronous=OFF;")
        self.write_conn.execute("PRAGMA temp_store=MEMORY;")
        self._log_and_print(f"Connected to DB: {self.db_path}", level="info")

    def get_pending_rows(self, last_id: int):
        cursor = self.read_conn.cursor()
        sql = (
            f"SELECT i.id, i.path FROM images i "
            f"LEFT JOIN {self.table_name} v ON i.id = v.image_id "
            f"WHERE v.{self.vector_column} IS NULL AND i.id > ? "
            f"ORDER BY i.id ASC LIMIT ?"
        )
        return cursor.execute(sql, (last_id, self.batch_size)).fetchall()


    def compute_vectors(self, paths: list[str]):
        """
        Muss von Unterklassen implementiert werden.
        Soll für jeden Pfad einen Vektor (oder None) zurückgegeben.
        """
        raise NotImplementedError

    def write_updates(self, id_vec_pairs: list[tuple[int, object]]):
        if not id_vec_pairs:
            self._log_and_print("No vectors to write for this batch.", level="warning")
            return

        blobs = []
        for rec_id, vec in id_vec_pairs:
            blob = sqlite3.Binary(pickle.dumps(vec, protocol=pickle.HIGHEST_PROTOCOL))
            blobs.append((rec_id, blob))

        cursor = self.write_conn.cursor()
        sql = (
            f"INSERT INTO {self.table_name} (image_id, {self.vector_column}) "
            f"VALUES (?, ?) "
            f"ON CONFLICT(image_id) DO UPDATE SET {self.vector_column}=excluded.{self.vector_column}"
        )

        try:
            self._log_and_print(f"Writing {len(blobs)} vectors to DB...", level="info")
            cursor.execute("BEGIN TRANSACTION;")
            cursor.executemany(sql, blobs)
            self.write_conn.commit()
            self._log_and_print(f"✅ Wrote {len(blobs)} vectors to DB.", level="info")
        except Exception as e:
            self.write_conn.rollback()
            self._log_and_print(f"❌ Write failed, rolled back: {e}", level="error")

    def batch_iterator(self):
        last_id = 0
        while True:
            rows = self.get_pending_rows(last_id)
            if not rows:
                break
            ids, paths = zip(*rows)
            yield ids, paths
            last_id = ids[-1]

    def run(self):
        total = self.read_conn.cursor().execute(
            f"SELECT COUNT(*) "
            f"FROM images i "
            f"LEFT JOIN {self.table_name} v ON i.id = v.image_id "
            f"WHERE v.{self.vector_column} IS NULL"
        ).fetchone()[0]

        self._log_and_print(f"Starting indexing for {total} images…", level="info")

        processed = 0
        for ids, paths in self.batch_iterator():
            self._log_and_print(
                f"Batch: IDs {ids[0]}–{ids[-1]}, {len(ids)} images", level="info"
            )
            vectors = self.compute_vectors(list(paths))
            id_vec = [(rid, vec) for rid, vec in zip(ids, vectors) if vec is not None]
            self.write_updates(id_vec)
            processed += len(ids)
            self._log_and_print(f"Progress: {processed}/{total}", level="info")

        self._log_and_print("✅ Indexing finished.", level="info")
        self.read_conn.close()
        self.write_conn.close()

# --- top-level helpers für multiprocessing ---


def load_image(
    img_path,
    img_size=None,
    gray=False,
    normalize=True,
    antialias=True,
    use_cv2=False,
):
    img_path = Path(img_path)
    if not img_path.exists():
        print(f"⚠️ Image not found: {img_path}")
        return None
    try:
        if use_cv2:
            import cv2
            flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
            img = cv2.imread(str(img_path), flag)
            if img is None:
                print(f"⚠️ Image not readable with cv2: {img_path}")
                return None
            if not gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img_size is not None:
                interp = cv2.INTER_LANCZOS4 if antialias else cv2.INTER_NEAREST
                img = cv2.resize(img, img_size, interpolation=interp)
            img = img.astype(np.float32)
            if normalize:
                img /= 255.0
            return img
        else:
            img = Image.open(img_path)
            if img.mode == "P":
                if "transparency" in img.info or img.info.get("transparency") is not None:
                    img = img.convert("RGBA")
                else:
                    img = img.convert("RGB")
            elif gray:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            if img_size is not None:
                resample = (
                    Image.Resampling.LANCZOS if antialias else Image.Resampling.NEAREST
                )
                img = img.resize(img_size, resample=resample)
            return img
    except Exception as e:
        print(f"⚠️ Error reading {img_path}: {e}")
        return None

