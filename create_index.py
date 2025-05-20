import os
import sqlite3
import pickle
import numpy as np
import faiss
import logging
from pathlib import Path

# FAISS multithreading workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FAISSIndexBuilderDB:
    def __init__(
        self,
        db_path: str = "images.db",
        vector_types: list = None,
        batch_size: int = 8192,
        index_file: str = None,
        hnsw_M: int = 32,
        efConstruction: int = 200,
        efSearch: int = 64,
        log_file: str = "faiss_builder.log",
        log_dir: str = "logs",
    ):  
        # Logging setup
        self.log_dir = log_dir
        self.log_file = log_file
        self._setup_logging()

        self.db_path = db_path
        self.vector_types = vector_types or ["clip", "color", "lpips"]
        self.vector_cols = [f"{t}_vector_blob" for t in self.vector_types]
        self.batch_size = batch_size

        name = "_".join(self.vector_types)
        self.index_file = Path(index_file) if index_file else Path(f"index_hnsw_{name}.faiss")

        self.hnsw_M = hnsw_M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        self.offset_table = f"faiss_index_offsets_{name}"

        self.read_conn = sqlite3.connect(self.db_path)
        self._configure_db(self.read_conn)
        self.read_cur = self.read_conn.cursor()

        self.write_conn = sqlite3.connect(self.db_path)
        self._configure_db(self.write_conn)
        self.write_cur = self.write_conn.cursor()

        self._prepare_offset_table()

    def _setup_logging(self):
        """
        Initializes logging to file in specified directory.
        """
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        full_path = Path(self.log_dir) / self.log_file
        logging.basicConfig(
            level=logging.INFO,
            filename=str(full_path),
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
            encoding="utf-8",
        )
        self._log(f"Logging initialized (file={full_path})", level="info")

    def _log(self, message: str, level: str = "info"):
        """
        Prints and logs a message at the specified level.
        """
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

    def _configure_db(self, conn):
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")

    def _prepare_offset_table(self):
        self.write_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.offset_table} (
                image_id INTEGER PRIMARY KEY,
                offset   INTEGER
            );
            """
        )
        self.write_conn.commit()
        self._log(f"Offset table '{self.offset_table}' is ready.", level="info")

    def _count_records(self):
        where_clause = " AND ".join(f"{col} IS NOT NULL" for col in self.vector_cols)
        result = self.read_cur.execute(
            f"SELECT COUNT(*) FROM images WHERE {where_clause}"
        ).fetchone()
        return result[0]

    def _batch_records(self):
        where_clause = " AND ".join(f"{col} IS NOT NULL" for col in self.vector_cols)
        select_cols = ", ".join(["id"] + self.vector_cols)
        query = f"SELECT {select_cols} FROM images WHERE {where_clause}"
        self.read_cur.execute(query)
        while True:
            rows = self.read_cur.fetchmany(self.batch_size)
            if not rows:
                break
            yield rows

    def _process_batch(self, rows):
        ids, embeddings = [], []
        for rec_id, *blobs in rows:
            parts = []
            skip = False
            for vt, blob in zip(self.vector_types, blobs):
                try:
                    vec = pickle.loads(blob)
                    if hasattr(vec, "cpu"):
                        vec = vec.cpu().numpy()
                    vec = np.asarray(vec, dtype="float16").ravel()
                    parts.append(vec)
                except Exception as e:
                    self._log(f"ID {rec_id}: error loading {vt}: {e}", level="warning")
                    skip = True
                    break
            if skip:
                continue
            ids.append(rec_id)
            embeddings.append(np.concatenate(parts))
        return ids, embeddings

    def _initialize_index(self, dim):
        index = faiss.IndexHNSWFlat(dim, self.hnsw_M)
        index.hnsw.efConstruction = self.efConstruction
        index.hnsw.efSearch = self.efSearch
        self._log(f"Created HNSW index (dim={dim}, M={self.hnsw_M})", level="info")
        return index

    def _store_offsets(self, ids, start_offset):
        pairs = [(rid, start_offset + i) for i, rid in enumerate(ids)]
        self.write_cur.executemany(
            f"INSERT OR REPLACE INTO {self.offset_table} (image_id, offset) VALUES (?, ?)",
            pairs,
        )
        self.write_conn.commit()

    def build_index(self, update_index: bool = False):
        combo = "_".join(self.vector_types)
        self._log(f"Starting FAISS index build for [{combo}]…", level="info")

        if not update_index:
            if self.index_file.exists():
                self._log(f"Removing existing index {self.index_file}", level="info")
                self.index_file.unlink()
            self._log(f"Clearing offset table {self.offset_table}", level="info")
            self.write_cur.execute(f"DELETE FROM {self.offset_table}")
            self.write_conn.commit()

        total = self._count_records()
        self._log(f"{total} complete records found.", level="info")
        if total == 0:
            self._log("No complete embeddings found; aborting.", level="error")
            return

        index = None
        offset_counter = 0
        batch_num = 0

        for batch in self._batch_records():
            batch_num += 1
            ids, embeddings = self._process_batch(batch)
            if not embeddings:
                continue

            arr = np.stack(embeddings).astype("float32")
            if index is None:
                index = self._initialize_index(arr.shape[1])

            index.add(arr)
            self._store_offsets(ids, offset_counter)
            offset_counter += len(ids)
            self._log(
                f"Batch {batch_num}: added {len(ids)} vectors (total {offset_counter}).",
                level="info",
            )

        self._log(f"Writing FAISS index to {self.index_file.resolve()}", level="info")
        faiss.write_index(index, str(self.index_file))
        self._log(f"Index saved ({index.ntotal} vectors).", level="info")

        self.read_conn.close()
        self.write_conn.close()
        self._log("Done.", level="info")


if __name__ == "__main__":
    builder = FAISSIndexBuilderDB(
        db_path="images.db",
        vector_types=["color", "hog"],  # or any combination of ["clip", "color", "lpips", "dreamsim"]
        batch_size=1024,
        hnsw_M=32,
        efConstruction=200,
        efSearch=50,
    )
    builder.build_index(update_index=False)
