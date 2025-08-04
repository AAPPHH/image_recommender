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
        Opens a SQLite connection using URI syntax and applies PRAGMA optimizations.

        Returns:
            sqlite3.Connection: Configured connection to the database.
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

    def create_tables(self):
        """
        Creates necessary tables in the database if they do not already exist:

        - `images`: Stores unique relative image paths.
        - `color_vectors`, `sift_vectors`, `dreamsim_vectors`: 
        One table per embedding type, linked via foreign keys to `images`.
        """
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS color_vectors (
                    image_id INTEGER PRIMARY KEY,
                    color_vector_blob BLOB,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS sift_vectors (
                    image_id INTEGER PRIMARY KEY,
                    sift_vector_blob BLOB,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS dreamsim_vectors (
                    image_id INTEGER PRIMARY KEY,
                    dreamsim_vector_blob BLOB,
                    FOREIGN KEY(image_id) REFERENCES images(id) ON DELETE CASCADE
                );
            """)
        print(f"Tables created in {self.db_path}.")

    def _batch_generator(self):
        """
        Generator that yields batches of relative image paths in POSIX format.

        Only files with the extensions `.jpg`, `.jpeg`, `.png` are considered.

        Yields:
            List[str]: A batch of relative file paths.
        """
        exts = (".jpg", ".jpeg", ".png")
        base = Path(self.base_folder)
        all_imgs = [p.relative_to(base).as_posix() for ext in exts for p in base.rglob(f"*{ext}")]
        batch = []
        for rel_path in all_imgs:
            batch.append(rel_path)  # already POSIX
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


    def process_batches(self): 
        """
        Main method to create tables and insert all image file paths into the `images` table in batches.

        For each batch:
            - Calls `INSERT OR IGNORE` to avoid duplicates.
            - Commits changes to the database after each batch.
        """
        self.create_tables()

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
    """
    CLI entry point: Initializes the ImageDBCreator with default paths and runs the batch processing.

    - Base folder: 'images_v3'
    - Database path: 'images.db'
    - Batch size: 10_000
    """
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