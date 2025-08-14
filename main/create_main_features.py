import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vector_scripts.create_color_vector import ColorVectorIndexer
from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer



def main():
    """
    Executes the full vector indexing pipeline for images stored in the current working directory.

    Steps:
        1. Computes color histogram vectors and stores them in the `color_vectors` table.
        2. Computes SIFT-VLAD vectors and stores them in the `sift_vectors` table.
        3. Computes DreamSim embeddings and stores them in the `dreamsim_vectors` table.

    Logging:
        Each vectorizer writes logs to its own file in the `logs/` directory.

    Assumes:
        - An SQLite database `images.db` exists and contains image file references.
        - The current directory contains the images or points to their base path.
    """
    db_path = "images.db"
    base_dir = Path(os.getcwd())


    # 1) Color Histogramm Vektoren berechnen
    print("=== Starte Color Vector Indexing ===")
    color_indexer = ColorVectorIndexer(
        db_path=db_path,
        base_dir=base_dir,
        batch_size=4096,
        log_file="color_indexer.log",
        log_dir="logs",
    )
    color_indexer.run()

    # 2) SIFTVLAD Vektoren berechnen
    print("=== Starte SIFTVLAD Vector Indexing ===")
    indexer = SIFTVLADVectorIndexer(
        db_path,
        base_dir,
        log_file="sift_vlad_indexer.log",
        batch_size=4096,
    )
    indexer.run()

    # 3) DreamSim Vektoren berechnen
    print("=== Starte DreamSim Vector Indexing ===")
    dreamsim_indexer = DreamSimVectorIndexer(
        db_path=db_path,
        base_dir=base_dir,
        batch_size=4096,
        model_batch=128,
        log_file="dreamsim_indexer.log",
        log_dir="logs",
    )
    dreamsim_indexer.run()

    print("=== Alle Vektor-Indexierungen abgeschlossen ===")


if __name__ == "__main__":
    """
    Entry point for executing the image vector indexing pipeline.
    """
    main()