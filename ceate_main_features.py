import os
from pathlib import Path

from vector_scripts.create_color_vector import ColorVectorIndexer
from vector_scripts.create_sift_vector import SIFTVLADVectorIndexer
from vector_scripts.create_dreamsim_vector import DreamSimVectorIndexer



def main():
    db_path = "images.db"
    base_dir = Path(os.getcwd())


    # 1) Color Histogramm Vektoren berechnen
    print("=== Starte Color Vector Indexing ===")
    color_indexer = ColorVectorIndexer(
        db_path=db_path,
        base_dir=base_dir,
        batch_size=16384,
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
        batch_size=16384,
    )
    indexer.run()

    # 3) DreamSim Vektoren berechnen
    print("=== Starte DreamSim Vector Indexing ===")
    dreamsim_indexer = DreamSimVectorIndexer(
        db_path=db_path,
        base_dir=base_dir,
        batch_size=16384,
        model_batch=128,
        log_file="dreamsim_indexer.log",
        log_dir="logs",
    )
    dreamsim_indexer.run()

    print("=== Alle Vektor-Indexierungen abgeschlossen ===")


if __name__ == "__main__":
    main()