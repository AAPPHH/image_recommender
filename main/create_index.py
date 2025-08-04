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
        self.vector_types = vector_types
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
        Sets up the logging system to record log messages to a file.
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
        Logs and prints a message.

        Args:
            message (str): The message to log.
            level (str): Logging level ("info", "warning", "error", "debug").
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
        """
        Applies SQLite PRAGMA performance optimizations.

        Args:
            conn (sqlite3.Connection): Database connection.
        """
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=OFF;")

    def _prepare_offset_table(self):
        """
        Creates the FAISS offset tracking table if it does not exist.
        This table maps image IDs to their vector offset in the FAISS index.
        """
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

    def _make_select_and_joins(self):
        """
        Constructs SQL SELECT and JOIN clauses based on configured vector types.

        Returns:
            Tuple[str, str]: SELECT columns string, JOIN clauses string.
        """
        select_cols = ["i.id"]
        join_strs = []
        for vtype in self.vector_types:
            alias = vtype[0]  # z. B. 'c', 's', 'd'
            vtable = f"{vtype}_vectors"
            vcol = f"{vtype}_vector_blob"
            select_cols.append(f"{alias}.{vcol}")
            join_strs.append(f"JOIN {vtable} {alias} ON i.id = {alias}.image_id")
        return ", ".join(select_cols), " ".join(join_strs)

    def _count_records(self):
        """
        Counts the number of complete records (images with all required vectors).

        Returns:
            int: Number of records available for indexing.
        """
        _, join_strs = self._make_select_and_joins()
        query = f"SELECT COUNT(*) FROM images i {join_strs}"
        result = self.read_cur.execute(query).fetchone()
        return result[0]

    def _batch_records(self):
        """
        Generator that yields batches of records from the database.

        Yields:
            List[Tuple]: Each batch as a list of tuples (id, vec1, vec2, ...)
        """
        select_cols, join_strs = self._make_select_and_joins()
        query = f"SELECT {select_cols} FROM images i {join_strs}"
        self.read_cur.execute(query)
        while True:
            rows = self.read_cur.fetchmany(self.batch_size)
            if not rows:
                break
            yield rows

    def _process_batch(self, rows):
        """
        Deserializes vector blobs and concatenates embeddings per record.

        Args:
            rows (List[Tuple]): Database rows with image ID and vector blobs.

        Returns:
            Tuple[List[int], List[np.ndarray]]: Image IDs and processed vectors.
        """
        ids, embeddings = [], []
        for rec_id, *blobs in rows:
            parts = []
            skip = False
            for vt, blob in zip(self.vector_types, blobs):
                try:
                    vec = pickle.loads(blob)
                    if hasattr(vec, "cpu"):
                        vec = vec.cpu().numpy()
                    vec = np.asarray(vec, dtype="float32").ravel()
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
    
    def find_valid_m(self, dim, candidates=(64, 56, 48, 32, 28, 24, 16, 12, 8)):
        """
        Finds a valid value of 'm' that evenly divides the dimensionality.

        Args:
            dim (int): Vector dimensionality.
            candidates (Tuple[int]): Possible values for m.

        Returns:
            int: Valid m value or 1 as fallback.
        """
        for m in candidates:
            if dim % m == 0:
                return m
        return 1

    def _initialize_index(self, dim, use_pq=True):
        """
        Initializes a FAISS index (HNSW or IVFPQ over HNSW).

        Args:
            dim (int): Dimensionality of the input vectors.
            use_pq (bool): Whether to use IVFPQ compression.

        Returns:
            faiss.Index: The initialized FAISS index.
        """
        if use_pq:
            coarse_quantizer = faiss.IndexHNSWFlat(dim, self.hnsw_M)
            coarse_quantizer.hnsw.efConstruction = self.efConstruction
            coarse_quantizer.hnsw.efSearch = self.efSearch

            nlist = 2048
            m = self.find_valid_m(dim)
            nbits = 12
            pq_index = faiss.IndexIVFPQ(coarse_quantizer, dim, nlist, m, nbits)
            self._log(f"Created IndexHNSW2Level with IVFPQ (dim={dim}, nlist={nlist}, m={m}, nbits={nbits})", level="info")
            return pq_index
        else:
            hnsw_index = faiss.IndexHNSWFlat(dim, self.hnsw_M)
            hnsw_index.hnsw.efConstruction = self.efConstruction
            hnsw_index.hnsw.efSearch = self.efSearch
            self._log(f"Created HNSW-Flat index (dim={dim}, M={self.hnsw_M})", level="info")
            return hnsw_index

    def _store_offsets(self, ids, start_offset):
        """
        Inserts vector ID-to-offset mappings into the offset table.

        Args:
            ids (List[int]): Image IDs.
            start_offset (int): Starting index in FAISS.
        """
        pairs = [(rid, start_offset + i) for i, rid in enumerate(ids)]
        self.write_cur.executemany(
            f"INSERT OR REPLACE INTO {self.offset_table} (image_id, offset) VALUES (?, ?)",
            pairs,
        )
        self.write_conn.commit()

    def build_index(self, update_index: bool = False):
        """
        Builds and saves a FAISS index from vectors stored in the database.

        Args:
            update_index (bool): If True, appends to an existing index. Otherwise, rebuilds from scratch.

        Process:
            - Loads and deserializes vectors in batches.
            - Concatenates vector types if multiple are configured.
            - Trains index if required.
            - Adds vectors to FAISS index.
            - Saves index to disk.
            - Writes ID-to-offset mappings to the database.
        """
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

        train_samples = []
        for batch in self._batch_records():
            _, embeddings = self._process_batch(batch)
            if not embeddings:
                continue
            arr = np.stack(embeddings).astype("float32")
            train_samples.append(arr)
            if sum([a.shape[0] for a in train_samples]) > 1500_00000:
                break
        train_vecs = np.concatenate(train_samples, axis=0)[:1500_00000]

        index = self._initialize_index(train_vecs.shape[1])

        if hasattr(index, "train") and not index.is_trained:
            self._log("Training IVFPQ...", level="info")
            index.train(train_vecs)
            self._log("IVFPQ trained.", level="info")
  
        offset_counter = 0
        batch_num = 0

        for batch in self._batch_records():
            batch_num += 1
            ids, embeddings = self._process_batch(batch)
            if not embeddings:
                continue

            arr = np.stack(embeddings).astype("float32")
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
    """
    Example CLI entry point to build a FAISS index from 'images.db' using color vectors only.

    Update the `vector_types` argument to combine features (e.g., ["color", "sift", "dreamsim"]).
    """
    builder = FAISSIndexBuilderDB(
        db_path="images.db",
        vector_types=["color"],  # or any combination of ["clip", "color", "lpips", "dreamsim"]
        batch_size=8192,
        hnsw_M=32,
        efConstruction=200,
        efSearch=50,
    )
    builder.build_index(update_index=False)
