import signal
import sys
import sqlite3
import pickle
import logging
from pathlib import Path


class BaseVectorIndexer:
    """
    Gemeinsame Basisklasse für alle Vektor-Indexer.
    Unterklassen müssen implementieren:
      - vector_column
      - get_pending_rows
      - compute_vectors
    """
    vector_column: str = None
    batch_size: int = 1000

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

        # Logging setup
        self._setup_logging(log_file, log_dir)
        self._init_db()

        # Handle Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _setup_logging(self, log_file: str, log_dir: str):
        """
        Initialisiert das Logging-Modul und schreibt das Logfile in den Unterordner `log_dir`.
        Erzeugt das Verzeichnis bei Bedarf.
        """
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
        """
        Gibt die Nachricht auf der Konsole aus und loggt sie mit dem angegebenen Level.
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
        """
        Muss von Unterklassen implementiert werden.
        Soll eine Liste von (id, rel_path)-Tuples zurückgeben.
        """
        raise NotImplementedError

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
            blobs.append((blob, rec_id))

        cursor = self.write_conn.cursor()
        sql = f"UPDATE images SET {self.vector_column} = ? WHERE id = ?"

        try:
            self._log_and_print(f"Writing {len(blobs)} vectors to DB...", level="info")
            cursor.execute("BEGIN TRANSACTION;")
            cursor.executemany(sql, blobs)
            self.write_conn.commit()
            self._log_and_print(f"✅ Wrote {len(blobs)} vectors to DB.", level="info")
        except Exception as e:
            self.write_conn.rollback()
            self._log_and_print(f"❌ Write failed, rolled back: {e}", level="error")

    def run(self):
        total = self.read_conn.cursor().execute(
            f"SELECT COUNT(*) FROM images WHERE {self.vector_column} IS NULL"
        ).fetchone()[0]
        self._log_and_print(f"Starting indexing for {total} images…", level="info")

        processed = 0
        last_id = 0
        while True:
            rows = self.get_pending_rows(last_id)
            if not rows:
                break
            ids, paths = zip(*rows)
            last_id = ids[-1]
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