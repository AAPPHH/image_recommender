import os
import sqlite3


class ImageDBCreator:
    def __init__(self, db_path, base_folder, batch_size=8192):
        """
        Initializes the ImageDBCreator.

        :param db_path: Path to the SQLite database file (e.g., "images.db").
        :param base_folder: Base folder in which to search for images.
        :param batch_size: Number of files per batch.
        """
        self.db_path = db_path
        self.base_folder = base_folder
        self.batch_size = batch_size

    def create_table(self):
        """
        Creates the 'images' table in the database if it does not already exist.
        The table contains:
          - a unique file path (relative path),
          - a column for the vector (vector_blob) (NULL if not yet available),
          - and a column to store the FAISS index offset.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE,
                vector_blob BLOB,
                faiss_index_offset INTEGER
            );
        """
        )
        c.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_faiss_offset ON images(faiss_index_offset);
            """
        )
        conn.commit()
        conn.close()
        print(f"Table 'images' in {self.db_path} has been created or already exists.")

    def _batch_generator(self):
        """
        Generator that recursively walks through the base folder and yields image paths (jpg, jpeg, png)
        in batches of the defined size. The relative path is stored.
        """
        current_batch = []
        for root, _, files in os.walk(self.base_folder):
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_folder)
                    current_batch.append(rel_path)
                    if len(current_batch) >= self.batch_size:
                        yield current_batch
                        current_batch = []
        if current_batch:
            yield current_batch

    def process_batches(self):
        """
        Reads all image paths from the specified base folder in batches and inserts them into the database.
        The relative paths are stored in the 'filepath' column.
        The columns 'vector_blob' and 'faiss_index_offset' remain NULL initially but are updated
        later during the indexing process.
        """
        # Create table
        self.create_table()

        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        batch_count = 0
        for batch in self._batch_generator():
            entries = [(fp,) for fp in batch]
            c.executemany("INSERT OR IGNORE INTO images (filepath) VALUES (?)", entries)
            conn.commit()
            batch_count += 1
            print(f"Batch {batch_count}: Inserted {len(batch)} files.")

        conn.close()
        print("Database has been updated.")


if __name__ == "__main__":
    base_folder = r"C:\Users\jfham\OneDrive\Dokumente\Workstation_Clones\image_recomender\image_recommender\images_v3"
    db_path = "images.db"
    db_creator = ImageDBCreator(
        db_path=db_path, base_folder=base_folder, batch_size=10000
    )
    db_creator.process_batches()
    print("Database creation completed.")
