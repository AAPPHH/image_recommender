import os
import sqlite3
from pathlib import Path

class ImageDBCreator:
    def __init__(
        self,
        db_path,
        base_folder,
        batch_size=8192,
        timeout=30_000,
        journal_mode="WAL",
        synchronous="OFF",
    ):
        dbp = Path(db_path)
        if not dbp.is_absolute():
            dbp = Path.cwd() / dbp
        self.db_path = dbp.resolve()

        bf = Path(base_folder)
        if not bf.is_absolute():
            bf = Path.cwd() / bf
        self.base_folder = bf.resolve()

        self.batch_size = batch_size
        self.timeout = timeout
        self.journal_mode = journal_mode
        self.synchronous = synchronous


    def _connect(self):
        """
        Opens a SQLite connection with PRAGMA parameters via URI.
        """
        mode = "rwc"
        uri = (
            f"file:{self.db_path}?mode={mode}"
            f"&cache=shared"
            f"&_pragma=journal_mode({self.journal_mode})"
            f"&_pragma=synchronous={self.synchronous}"
            f"&_pragma=temp_store(MEMORY)"
            f"&_pragma=busy_timeout={self.timeout}"
        )
        timeout_sec = self.timeout / 1000
        return sqlite3.connect(uri, uri=True, timeout=timeout_sec)

    def create_table(self):
        """
        Creates the 'images' table with BLOB columns and an index.
        """
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    color_vector_blob BLOB,
                    lpips_vector_blob BLOB,
                    dreamsim_vector_blob BLOB
                );
                """
            )
        print(f"Table 'images' in {self.db_path} is ready.")

    def _batch_generator(self):
        """
        Walks through base_folder and yields batches of relative image paths.
        """
        batch = []
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full = Path(root) / file
                    rel = full.relative_to(self.base_folder)
                    batch.append(str(rel))
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
        if batch:
            yield batch

    def process_batches(self):
        self.create_table()

        with self._connect() as conn:
            c = conn.cursor()
            batch_num = 0
            for batch in self._batch_generator():
                entries = [(fp,) for fp in batch]
                c.executemany(
                    "INSERT OR IGNORE INTO images (path) VALUES (?)", entries
                )
                conn.commit()
                batch_num += 1
                print(f"Batch {batch_num}: inserted {len(batch)} paths.")
        print("All file paths saved to the database.")


if __name__ == "__main__":
    BASE_FOLDER = "images_v3"
    DB_PATH = "images.db"

    creator = ImageDBCreator(
        db_path=DB_PATH,
        base_folder=BASE_FOLDER,
        batch_size=10000,
        timeout=30000,
        journal_mode="WAL",
        synchronous="OFF",
    )
    creator.process_batches()
    print("Database creation completed.")
