#!/usr/bin/env python3
import os
import sqlite3
import pickle
import numpy as np
import faiss
from pathlib import Path

from logging_utils import setup_logging, console_and_log

# FAISS multithreading workaround
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize logging
setup_logging("faiss_builder.log", log_dir="logs")


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
    ):
        self.db_path = db_path
        self.vector_types = vector_types or ["clip", "color", "lpips"]
        self.vector_cols = [f"{t}_vector_blob" for t in self.vector_types]
        self.batch_size = batch_size

        # Index file
        name = "_".join(self.vector_types)
        self.index_file = Path(index_file) if index_file else Path(f"index_hnsw_{name}.faiss")

        self.hnsw_M = hnsw_M
        self.efConstruction = efConstruction
        self.efSearch = efSearch

        # Offset table name for this combination
        self.offset_table = f"faiss_index_offsets_{name}"

        # Database connections
        self.read_conn = sqlite3.connect(self.db_path)
        self.read_conn.execute("PRAGMA journal_mode=WAL;")
        self.read_conn.execute("PRAGMA synchronous=OFF;")
        self.read_cur = self.read_conn.cursor()

        self.write_conn = sqlite3.connect(self.db_path)
        self.write_conn.execute("PRAGMA journal_mode=WAL;")
        self.write_conn.execute("PRAGMA synchronous=OFF;")
        self.write_cur = self.write_conn.cursor()

        # Create offset table if not exists
        self.write_cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.offset_table} (
                image_id INTEGER PRIMARY KEY,
                offset   INTEGER
            );
            """
        )
        self.write_conn.commit()
        console_and_log(f"Offset table '{self.offset_table}' is ready.", level="info")

    def build_index(self, update_index: bool = False):
        combo = "_".join(self.vector_types)
        console_and_log(f"Starting FAISS index build for [{combo}]…", level="info")

        # Remove old data if building fresh
        if not update_index:
            if self.index_file.exists():
                console_and_log(f"Removing existing index {self.index_file}", level="info")
                self.index_file.unlink()
            console_and_log(f"Clearing offset table {self.offset_table}", level="info")
            self.write_cur.execute(f"DELETE FROM {self.offset_table}")
            self.write_conn.commit()

        # Only select records with all required vectors
        where_clause = " AND ".join(f"{col} IS NOT NULL" for col in self.vector_cols)
        total = self.read_cur.execute(
            f"SELECT COUNT(*) FROM images WHERE {where_clause}"
        ).fetchone()[0]
        console_and_log(f"{total} complete records found.", level="info")
        if total == 0:
            console_and_log("No complete embeddings found; aborting.", level="error")
            return

        select_cols = ", ".join(["id"] + self.vector_cols)
        self.read_cur.execute(f"SELECT {select_cols} FROM images WHERE {where_clause}")

        index = None
        current_offset = 0
        batch_num = 0

        while True:
            rows = self.read_cur.fetchmany(self.batch_size)
            if not rows:
                break
            batch_num += 1

            ids, embeddings = [], []
            for rec_id, *blobs in rows:
                parts = []
                ok = True
                for vt, blob in zip(self.vector_types, blobs):
                    if blob is None:
                        ok = False
                        console_and_log(f"ID {rec_id}: {vt}-blob is NULL, skipping.", level="warning")
                        break
                    try:
                        vec = pickle.loads(blob)
                        if hasattr(vec, "cpu"):
                            vec = vec.cpu().numpy()
                        vec = np.asarray(vec, dtype="float32").ravel()
                        parts.append(vec)
                    except Exception as e:
                        ok = False
                        console_and_log(f"ID {rec_id}: error loading {vt}: {e}", level="error")
                        break

                if not ok:
                    continue

                ids.append(rec_id)
                embeddings.append(np.concatenate(parts))

            if not embeddings:
                continue

            arr = np.stack(embeddings).astype("float32")

            if index is None:
                dim = arr.shape[1]
                index = faiss.IndexHNSWFlat(dim, self.hnsw_M)
                index.hnsw.efConstruction = self.efConstruction
                index.hnsw.efSearch = self.efSearch
                console_and_log(f"Created HNSW index (dim={dim}, M={self.hnsw_M})", level="info")

            index.add(arr)

            # Store offsets in the DB
            pairs = [(rid, current_offset + i) for i, rid in enumerate(ids)]
            self.write_cur.executemany(
                f"INSERT OR REPLACE INTO {self.offset_table} (image_id, offset) VALUES (?, ?)",
                pairs,
            )
            self.write_conn.commit()

            current_offset += len(ids)
            console_and_log(
                f"Batch {batch_num}: added {len(ids)} vectors (total {current_offset}).",
                level="info",
            )

        # Write the FAISS index to disk
        console_and_log(f"Writing FAISS index to {self.index_file.resolve()}", level="info")
        faiss.write_index(index, str(self.index_file))
        console_and_log(f"Index saved ({index.ntotal} vectors).", level="info")

        # Clean up
        self.read_conn.close()
        self.write_conn.close()
        console_and_log("Done.", level="info")


if __name__ == "__main__":
    builder = FAISSIndexBuilderDB(
        db_path="images.db",
        vector_types=["lpips", "dreamsim"],  # or any combination of ["clip", "color", "lpips", "dreamsim"]
        batch_size=1024,
        hnsw_M=32,
        efConstruction=200,
        efSearch=50,
    )
    builder.build_index(update_index=False)
