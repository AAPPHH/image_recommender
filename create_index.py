import os
import sqlite3
import pickle
import numpy as np
import faiss
import logging
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


class FAISSIndexBuilderDB:
    def __init__(
        self,
        db_path="images.db",
        batch_size=1024,
        index_file="index_hnsw.faiss",
        hnsw_M=32,
        efConstruction=200,
        efSearch=50,
    ):
        """
        Initializes the FAISSIndexBuilder, which builds an HNSW index from embeddings
        stored in the SQLite database (table "images"). The FAISS index offset is saved
        in the "faiss_index_offset" column of the "images" table.

        :param db_path: Path to the SQLite database.
        :param batch_size: Number of records per batch.
        :param index_file: Filename where the FAISS index will be saved.
        :param hnsw_M: Number of connections per vector (HNSW parameter).
        :param efConstruction: Parameter for the construction phase of the HNSW index.
        :param efSearch: Parameter that influences search accuracy and time.
        """
        self.db_path = db_path
        self.batch_size = batch_size
        self.index_file = index_file
        self.hnsw_M = hnsw_M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        self.conn = sqlite3.connect(self.db_path)
        self.read_cursor = self.conn.cursor()
        self.write_cursor = self.conn.cursor()
        self.conn.commit()

    def _generate_batches(self):
        """
        Generator that yields records from the "images" table in batches.
        Only entries where vector_blob is not NULL are considered.

        :yield: Tuple (list of record IDs, list of embeddings, list of file paths)
        """
        self.read_cursor.execute(
            "SELECT id, filepath, vector_blob FROM images WHERE vector_blob IS NOT NULL"
        )
        while True:
            rows = self.read_cursor.fetchmany(self.batch_size)
            if not rows:
                break
            batch_ids = []
            batch_embeddings = []
            batch_paths = []
            for rec_id, filepath, blob in rows:
                try:
                    emb = pickle.loads(blob)
                    if hasattr(emb, "cpu"):
                        emb = emb.cpu().numpy()
                except Exception as e:
                    logging.error(f"Error deserializing {filepath}: {e}")
                    continue
                batch_ids.append(rec_id)
                batch_embeddings.append(emb)
                batch_paths.append(filepath)
            yield batch_ids, batch_embeddings, batch_paths

    def build_index(self, update_index=False):
        """
        Builds a FAISS HNSW index from the embeddings stored in the SQLite database.
        Vectors are added to the index in batches, and each record's FAISS index offset
        is stored in the "faiss_index_offset" column of the "images" table.

        :param update_index: If True, new vectors are added to an existing index.
                             Otherwise, a new index is created and offsets are reset.
        """
        if not update_index:
            logging.info(
                "Clearing 'faiss_index_offset' column because a new index is being created."
            )
            self.write_cursor.execute("UPDATE images SET faiss_index_offset = NULL")
            self.conn.commit()

        index = None
        if update_index and os.path.exists(self.index_file):
            index = faiss.read_index(self.index_file)
            logging.info(f"Loaded existing HNSW index with {index.ntotal} vectors.")

        batch_count = 0
        current_offset = index.ntotal if (index is not None and update_index) else 0

        for batch_ids, batch_embeddings, batch_paths in self._generate_batches():
            batch_count += 1
            batch_embeddings_np = np.array(batch_embeddings, dtype="float32")
            if index is None:
                dim = batch_embeddings_np.shape[1]
                index = faiss.IndexHNSWFlat(dim, self.hnsw_M)
                index.hnsw.efConstruction = self.efConstruction
                index.hnsw.efSearch = self.efSearch
                logging.info(
                    f"Created new HNSW index with dimension {dim}, M={self.hnsw_M}, "
                    f"efConstruction={self.efConstruction}, efSearch={self.efSearch}."
                )

            index.add(batch_embeddings_np)

            for i, rec_id in enumerate(batch_ids):
                offset = current_offset + i
                self.write_cursor.execute(
                    "UPDATE images SET faiss_index_offset = ? WHERE id = ?",
                    (offset, rec_id),
                )
            self.conn.commit()

            current_offset += len(batch_ids)
            logging.info(
                f"Processed batch {batch_count}: added {len(batch_ids)} vectors. Total vectors: {index.ntotal}."
            )

        if index is None:
            logging.error("No embeddings found. Indexing aborted.")
            return

        faiss.write_index(index, self.index_file)
        logging.info(f"HNSW index saved to '{self.index_file}'.")
        logging.info(
            "The 'faiss_index_offset' column has been updated in the 'images' table."
        )


if __name__ == "__main__":
    builder = FAISSIndexBuilderDB(
        db_path="images.db",
        batch_size=8192,
        index_file="index_hnsw.faiss",
        hnsw_M=32,
        efConstruction=200,
        efSearch=50,
    )
    builder.build_index(update_index=False)
